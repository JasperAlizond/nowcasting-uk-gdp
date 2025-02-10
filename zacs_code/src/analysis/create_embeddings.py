"""
Text Embeddings Generator Module optimized for Apple Silicon (M1/M2/M3) Macs with limited memory
"""

import time
from typing import List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import gc

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Statistics from the embedding generation process"""

    total_time: float
    avg_time_per_item: float
    total_items: int
    num_batches: int
    output_shape: Tuple[int, int]
    peak_memory_mb: float


class EmbeddingsGenerator:
    """Handles the generation of embeddings optimized for Apple Silicon"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: Optional[int] = None,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize the embeddings generator

        Args:
            model_name: Name of the transformer model to use
            batch_size: Number of items to process at once (None for auto-detect)
            max_length: Maximum token length for text
            device: Device to use for computation (None for auto-detect)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._setup_device(device)

        # Auto-configure batch size based on available memory
        self.batch_size = self._configure_batch_size(batch_size)

        self.model, self.tokenizer = self._load_model_and_tokenizer()
        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Model {model_name} loaded successfully")

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup the computation device optimized for Apple Silicon"""
        if device is None:
            # Check for MPS (Metal Performance Shaders) availability
            if torch.backends.mps.is_available():
                device = "mps"  # Use Metal for Apple Silicon
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        return torch.device(device)

    def _configure_batch_size(self, batch_size: Optional[int]) -> int:
        """Configure optimal batch size based on available memory"""
        if batch_size is not None:
            return batch_size

        # For 8GB M3 Macs, we need to be conservative with memory
        if self.device == torch.device("mps"):
            return 8  # Conservative batch size for 8GB memory
        elif self.device == torch.device("cuda"):
            return 32
        else:
            return 16  # CPU processing

    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load and configure the model and tokenizer with memory optimization"""
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, model_max_length=self.max_length
            )

            # Load model with memory optimization
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for better compatibility
                low_cpu_mem_usage=True,
            ).to(self.device)

            model.eval()  # Set to evaluation mode
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _preprocess_texts(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """Preprocess input texts with memory-efficient handling"""
        processed = []
        for text in texts:
            # Process one text at a time to avoid memory spikes
            processed.append(str(text) if pd.notna(text) else "")
        return processed

    def _create_batches(self, texts: List[str]) -> List[List[str]]:
        """Split texts into batches with memory considerations"""
        return [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

    def _process_batch(self, batch: List[str]) -> np.ndarray:
        """Process a single batch of texts with memory optimization"""
        try:
            # Tokenize and move to device
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**encoded)

            # Generate embeddings through mean pooling
            attention_mask = encoded["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            # Calculate mean pooling
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().numpy()

            # Clear unnecessary tensors
            del encoded, outputs, token_embeddings, input_mask_expanded
            if self.device == torch.device("mps"):
                torch.mps.empty_cache()
            elif self.device == torch.device("cuda"):
                torch.cuda.empty_cache()

            return embeddings_np

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    def generate_embeddings(
        self, df: pd.DataFrame, text_column: str = "abstract"
    ) -> Tuple[pd.DataFrame, EmbeddingStats]:
        """Generate embeddings with memory-efficient processing"""
        logger.info("Starting embeddings generation process...")

        # Prepare texts
        texts = self._preprocess_texts(df[text_column])
        total_texts = len(texts)
        batches = self._create_batches(texts)

        logger.info(f"Total texts to process: {total_texts}")
        logger.info(
            f"Number of batches: {len(batches)} (batch size: {self.batch_size})"
        )

        # Generate embeddings
        embeddings_list = []
        start_time = time.time()
        peak_memory = 0

        with torch.no_grad():
            for batch in tqdm(batches, desc="Processing batches"):
                # Process batch
                batch_embeddings = self._process_batch(batch)
                embeddings_list.append(batch_embeddings)

                # Track memory usage
                if self.device == torch.device("mps"):
                    # Estimate memory usage for MPS
                    memory_used = len(batch_embeddings.tobytes()) / (1024 * 1024)
                elif self.device == torch.device("cuda"):
                    memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    memory_used = 0

                peak_memory = max(peak_memory, memory_used)

                # Force garbage collection
                gc.collect()

        # Combine embeddings
        all_embeddings = np.vstack(embeddings_list)
        embeddings_df = pd.DataFrame(
            all_embeddings,
            columns=[f"embedding_{i}" for i in range(all_embeddings.shape[1])],
        )

        # Create final DataFrame
        final_df = pd.concat([df, embeddings_df], axis=1)

        # Calculate statistics
        end_time = time.time()
        total_time = end_time - start_time

        stats = EmbeddingStats(
            total_time=total_time,
            avg_time_per_item=total_time / total_texts,
            total_items=total_texts,
            num_batches=len(batches),
            output_shape=final_df.shape,
            peak_memory_mb=peak_memory,
        )

        self._log_statistics(stats)
        self._cleanup()

        return final_df, stats

    def _log_statistics(self, stats: EmbeddingStats):
        """Log processing statistics"""
        logger.info("\nProcessing Statistics:")
        logger.info(f"Total processing time: {stats.total_time:.2f} seconds")
        logger.info(f"Average time per item: {stats.avg_time_per_item:.2f} seconds")
        logger.info(f"Total items processed: {stats.total_items}")
        logger.info(f"Number of batches: {stats.num_batches}")
        logger.info(f"Final output shape: {stats.output_shape}")
        logger.info(f"Peak memory usage: {stats.peak_memory_mb:.2f} MB")

    def _cleanup(self):
        """Cleanup after processing"""
        gc.collect()
        if self.device == torch.device("mps"):
            torch.mps.empty_cache()
        elif self.device == torch.device("cuda"):
            torch.cuda.empty_cache()


def load_data(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data from CSV files"""
    data_dir = Path(data_dir)

    try:
        train_path = data_dir / "train_merged.csv"
        test_path = data_dir / "test_merged.csv"

        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)

        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def save_processed_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Union[str, Path]
) -> None:
    """Save processed DataFrames to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_path = output_dir / "train_embeddings.csv"
        test_path = output_dir / "test_embeddings.csv"

        logger.info(f"Saving processed training data to {train_path}")
        train_df.to_csv(train_path, index=False)

        logger.info(f"Saving processed test data to {test_path}")
        test_df.to_csv(test_path, index=False)

        logger.info("Data saved successfully")

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise


def process_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "abstract",
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process both training and test datasets"""
    generator = EmbeddingsGenerator(**kwargs)

    logger.info("Processing training data...")
    train_final, train_stats = generator.generate_embeddings(train_df, text_column)

    logger.info("\nProcessing test data...")
    test_final, test_stats = generator.generate_embeddings(test_df, text_column)

    return train_final, test_final


if __name__ == "__main__":
    # Set up directories
    data_dir = Path("data")
    processed_dir = data_dir / "processed"

    # Load data
    logger.info("Loading data...")
    train_df, test_df = load_data(data_dir)

    # Process datasets with M3-optimized settings
    logger.info("Generating embeddings...")
    train_final, test_final = process_datasets(
        train_df,
        test_df,
        batch_size=None,  # Will auto-configure based on available memory
        max_length=512,
    )

    # Save processed data
    logger.info("Saving processed data...")
    save_processed_data(train_final, test_final, processed_dir)
