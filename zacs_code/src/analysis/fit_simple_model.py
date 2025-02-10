import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import logging
import joblib
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimplePCARandomForest:
    def __init__(self, max_features: int = 50, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target from dataframe"""
        # Get numeric columns excluding target and date
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ["y", "date"]]

        # Filter out rows with NaN values in any numeric column
        df = df.dropna(subset=feature_cols + (["y"] if "y" in df.columns else []))

        X = df[feature_cols].values
        y = df["y"].values if "y" in df.columns else None

        return X, y

    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA to reduce dimensions to max_features"""
        if self.pca is None:
            self.pca = PCA(n_components=self.max_features)
            return self.pca.fit_transform(X)
        return self.pca.transform(X)

    def process_dataframe(
        self, df: pd.DataFrame, is_training: bool = False
    ) -> pd.DataFrame:
        """Process dataframe with scaling and PCA, return quarterly averaged dataframe"""
        # Ensure Date column is datetime
        df = df.copy()
        df["Date"] = pd.to_datetime(df["date"])

        # Prepare features
        X, y = self.prepare_data(df)

        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Apply PCA
        X_reduced = self._reduce_dimensions(X_scaled)

        # Create new dataframe with reduced features
        if X_reduced.shape[0] != df.shape[0]:
            logger.warning(
                f"Shape mismatch: df={df.shape[0]}, X_reduced={X_reduced.shape[0]}"
            )
            df = df.iloc[: X_reduced.shape[0]]  # Trim df index to match X_reduced

        reduced_df = pd.DataFrame(
            X_reduced,
            columns=[f"pca_{i + 1}" for i in range(X_reduced.shape[1])],
            index=df.index,
        )

        # Add back Date and target
        reduced_df["Date"] = df["Date"]
        if y is not None:
            reduced_df["y"] = y

        # Group by quarter instead of month
        quarterly_df = (
            reduced_df.groupby(pd.Grouper(key="Date", freq="Q")).mean().reset_index()
        )

        return quarterly_df

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Train model and process data"""
        logger.info("Processing training data...")
        train_processed = self.process_dataframe(train_df, is_training=True)

        logger.info("Processing test data...")
        test_processed = self.process_dataframe(test_df, is_training=False)

        # Save processed dataframes
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)

        train_processed.to_csv(output_dir / "train_processed_monthly.csv", index=False)
        test_processed.to_csv(output_dir / "test_processed_monthly.csv", index=False)

        # Prepare training data
        X_train = train_processed.drop(["Date", "y"], axis=1).values
        y_train = train_processed["y"].values

        # Train Random Forest
        logger.info("Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(
            test_processed.drop(["Date", "y"], axis=1).values
        )

        metrics = {
            "train_mse": mean_squared_error(y_train, train_pred),
            "test_mse": mean_squared_error(test_processed["y"], test_pred),
            "train_r2": r2_score(y_train, train_pred),
            "test_r2": r2_score(test_processed["y"], test_pred),
        }

        self._log_metrics(metrics)

        return {
            "model": self.model,
            "scaler": self.scaler,
            "pca": self.pca,
            "metrics": metrics,
            "train_processed": train_processed,
            "test_processed": test_processed,
        }

    def _log_metrics(self, metrics: Dict):
        """Log model performance metrics"""
        logger.info("\nModel Performance:")
        logger.info(f"Training MSE: {metrics['train_mse']:.4f}")
        logger.info(f"Test MSE: {metrics['test_mse']:.4f}")
        logger.info(f"Training R²: {metrics['train_r2']:.4f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")

    def save_model(self, model_data: Dict, filename: str):
        """Save model and associated data"""
        output_dir = Path("data/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / filename

        save_dict = {
            "model": model_data["model"],
            "scaler": model_data["scaler"],
            "pca": model_data["pca"],
            "metrics": model_data["metrics"],
        }

        logger.info(f"Saving model to {save_path}")
        joblib.dump(save_dict, save_path)

    @staticmethod
    def load_model(model_path: str) -> Dict:
        """Load saved model and associated data"""
        logger.info(f"Loading model from {model_path}")
        return joblib.load(model_path)


if __name__ == "__main__":
    # Load data
    data_dir = Path("data")
    train_df = pd.read_csv(data_dir / "processed" / "train_embeddings.csv")
    test_df = pd.read_csv(data_dir / "processed" / "test_embeddings.csv")

    # Initialize and train model
    trainer = SimplePCARandomForest(max_features=50)
    results = trainer.train(train_df, test_df)

    # Save model
    trainer.save_model(results, "gdp_prediction_model.joblib")

    logger.info("Model training complete!")
    logger.info("Processed data saved to data/processed/")
    logger.info("Model saved to data/models/")
