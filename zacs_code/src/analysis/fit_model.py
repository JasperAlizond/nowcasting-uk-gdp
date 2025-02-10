import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Union, List
import gc
from dataclasses import dataclass
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure matplotlib for Apple Silicon
plt.rcParams["figure.dpi"] = 150  # Higher DPI for Retina displays


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""

    train_mse: float
    test_mse: float
    train_r2: float
    test_r2: float
    n_features: int
    n_components: int
    variance_explained: float


class MemoryEfficientPCA:
    """Memory-efficient PCA implementation with progress tracking"""

    def __init__(
        self, n_components: Optional[int] = None, variance_threshold: float = 0.95
    ):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = None
        self.batch_size = 1000  # Adjust based on available memory

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data with memory efficiency"""
        logger.info("Starting PCA fit_transform...")

        # Initialize PCA
        if self.n_components is None:
            self.pca = PCA(n_components=self.variance_threshold, svd_solver="full")
        else:
            self.pca = PCA(n_components=self.n_components)

        # Process in batches if data is large
        if len(X) > self.batch_size:
            return self._batch_transform(X, fit=True)

        result = self.pca.fit_transform(X)
        gc.collect()  # Force garbage collection
        return result

    def transform(
        self, X: np.ndarray, dates: Optional[pd.Series] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transform data with memory efficiency and optional temporal aggregation

        Args:
            X: Input data array
            dates: Optional series of dates for temporal aggregation

        Returns:
            Transformed data, either as numpy array or DataFrame if dates provided
        """
        transformed_data = (
            self._batch_transform(X, fit=False)
            if len(X) > self.batch_size
            else self.pca.transform(X)
        )

        if dates is not None:
            return self._aggregate_by_month(transformed_data, dates)
        return transformed_data

    def _aggregate_by_month(self, data: np.ndarray, dates: pd.Series) -> pd.DataFrame:
        """Aggregate PCA features by month

        Args:
            data: Transformed PCA data
            dates: Series of dates corresponding to each row

        Returns:
            DataFrame with monthly averages of PCA features
        """
        # Create DataFrame with dates and PCA features
        df = pd.DataFrame(data, columns=[f"pca_{i + 1}" for i in range(data.shape[1])])
        df["date"] = pd.to_datetime(dates)

        # Resample to monthly frequency and compute means
        monthly_df = df.set_index("date").resample("M").mean().reset_index()

        # Save to processed directory
        output_path = Path("data/processed/monthly_pca_features.csv")
        monthly_df.to_csv(output_path, index=False)
        logger.info(f"Saved monthly PCA features to {output_path}")

    def _batch_transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Process data in batches with progress bar"""
        n_samples = len(X)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        results = []
        desc = "Fitting PCA" if fit else "Transforming data"
        for i in tqdm(range(n_batches), desc=desc, unit="batch"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch = X[start_idx:end_idx]

            if fit and i == 0:
                batch_result = self.pca.fit_transform(batch)
            else:
                batch_result = self.pca.transform(batch)

            results.append(batch_result)
            gc.collect()  # Force garbage collection

        return np.vstack(results)


class ModelTrainer:
    """Handles model training with memory optimization and progress tracking"""

    def __init__(
        self,
        reduction_method: str = "pca",
        n_components: Optional[int] = None,
        variance_threshold: float = 0.95,
        random_state: int = 42,
        output_dir: Optional[Union[str, Path]] = None,
        max_features: int = 50,  # New parameter to cap maximum features
    ):
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else None
        self.max_features = max_features

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_data(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training, handling NaN values and extracting features

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Make copies to avoid modifying original data
        train_df = train_df.copy()
        test_df = test_df.copy()

        # Remove rows with NaN target values
        train_df = train_df.dropna(subset=["y"])
        test_df = test_df.dropna(subset=["y"])

        logger.info(f"Training samples after removing NaN targets: {len(train_df)}")
        logger.info(f"Test samples after removing NaN targets: {len(test_df)}")

        # Convert column names to strings
        train_df.columns = train_df.columns.astype(str)
        test_df.columns = test_df.columns.astype(str)

        # Get numeric columns excluding target
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        embedding_cols = [col for col in numeric_cols if col != "y"]

        # Extract features and target
        X_train = train_df[embedding_cols].values
        y_train = train_df["y"].values
        X_test = test_df[embedding_cols].values
        y_test = test_df["y"].values

        return X_train, y_train, X_test, y_test

    def _scale_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Scale features with memory efficiency"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clear memory
        gc.collect()

        return X_train_scaled, X_test_scaled, scaler

    def _reduce_dimensions(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Union[MemoryEfficientPCA, np.ndarray]]:
        """Perform dimensionality reduction"""
        if self.reduction_method == "pca":
            # First try with variance threshold
            reducer = MemoryEfficientPCA(
                n_components=None,  # Start with variance threshold
                variance_threshold=self.variance_threshold,
            )
            X_train_reduced = reducer.fit_transform(X_train)

            # If we have too many components, force the number to max_features
            if X_train_reduced.shape[1] > self.max_features:
                logger.info(
                    f"Reducing from {X_train_reduced.shape[1]} to {self.max_features} components"
                )
                reducer = MemoryEfficientPCA(
                    n_components=self.max_features, variance_threshold=None
                )
                X_train_reduced = reducer.fit_transform(X_train)

            X_test_reduced = reducer.transform(X_test)
            if self.output_dir:
                self._plot_pca_results(reducer.pca)

        elif self.reduction_method == "feature_importance":
            X_train_reduced, X_test_reduced, selected_features = self._select_features(
                X_train, X_test
            )
            reducer = selected_features

        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")

        logger.info(f"Final number of features: {X_train_reduced.shape[1]}")
        return X_train_reduced, X_test_reduced, reducer

    def _plot_pca_results(self, pca: PCA):
        """Plot PCA results with memory efficiency"""
        if not self.output_dir:
            return

        plt.figure(figsize=(10, 4))

        # Plot cumulative explained variance
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), "bo-")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Explained Variance vs Components")

        # Plot individual contributions
        plt.subplot(1, 2, 2)
        plt.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Individual Component Contribution")

        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_analysis.png", dpi=150)
        plt.close()

    def _select_features(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select features based on importance with progress tracking"""
        logger.info("Training Random Forest for feature selection...")
        rf = RandomForestRegressor(
            n_estimators=50,  # Reduced for memory efficiency
            random_state=self.random_state,
            verbose=0,  # Disable built-in verbosity in favor of tqdm
        )

        with tqdm(total=50, desc="Training trees", unit="tree") as pbar:
            rf.fit(X_train, self.y_train)
            pbar.update(50)  # Update progress after fitting

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        if self.n_components is None:
            cumsum = np.cumsum(importances[indices])
            self.n_components = np.where(cumsum >= self.variance_threshold)[0][0] + 1

        selected_features = indices[: self.n_components]

        if self.output_dir:
            self._plot_feature_importance(importances, indices)

        return (
            X_train[:, selected_features],
            X_test[:, selected_features],
            selected_features,
        )

    def _plot_feature_importance(self, importances: np.ndarray, indices: np.ndarray):
        """Plot feature importance results"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(importances)), importances[indices])
        plt.xlabel("Feature Index")
        plt.ylabel("Feature Importance")
        plt.title("Feature Importance Distribution")

        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(importances[indices]), "bo-")
        plt.axhline(
            y=self.variance_threshold,
            color="r",
            linestyle="--",
            label=f"{self.variance_threshold * 100}% threshold",
        )
        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Importance")
        plt.title("Cumulative Feature Importance")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance.png", dpi=150)
        plt.close()

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Train model with memory optimization and progress tracking"""
        logger.info("Starting model training pipeline...")

        with tqdm(total=6, desc="Training progress", unit="step") as pbar:
            # Prepare data
            X_train, y_train, X_test, y_test = self._prepare_data(train_df, test_df)
            self.y_train = y_train  # Store for feature selection
            pbar.update(1)
            pbar.set_description("Data preparation complete")

            # Scale features
            logger.info("Scaling features...")
            X_train_scaled, X_test_scaled, scaler = self._scale_features(
                X_train, X_test
            )
            self.scaler = scaler  # Store scaler
            pbar.update(1)
            pbar.set_description("Feature scaling complete")

            # Reduce dimensions
            logger.info(
                f"Performing dimensionality reduction using {self.reduction_method}..."
            )
            X_train_reduced, X_test_reduced, reducer = self._reduce_dimensions(
                X_train_scaled, X_test_scaled
            )
            self.reducer = reducer  # Store reducer
            pbar.update(1)
            pbar.set_description("Dimensionality reduction complete")

            # Train model with optimized parameters
            logger.info("Training Random Forest...")
            rf = RandomForestRegressor(
                n_estimators=200,  # More trees for better performance
                max_depth=None,  # Allow full depth
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.5,  # Use half of features for splits
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
            )
            rf.fit(X_train_reduced, y_train)
            self.model = rf  # Store model
            pbar.update(1)
            pbar.set_description("Model training complete")

            # Make predictions
            train_pred = rf.predict(X_train_reduced)
            test_pred = rf.predict(X_test_reduced)
            pbar.update(1)
            pbar.set_description("Predictions complete")

            # Calculate metrics
            metrics = ModelMetrics(
                train_mse=mean_squared_error(y_train, train_pred),
                test_mse=mean_squared_error(y_test, test_pred),
                train_r2=r2_score(y_train, train_pred),
                test_r2=r2_score(y_test, test_pred),
                n_features=X_train.shape[1],
                n_components=X_train_reduced.shape[1],
                variance_explained=self.variance_threshold,
            )

            self._log_metrics(metrics)
            pbar.update(1)
            pbar.set_description("Training pipeline complete")

            if self.output_dir:
                self._plot_predictions(y_train, train_pred, y_test, test_pred)

            return {
                "scaler": scaler,
                "reducer": reducer,
                "model": rf,
                "metrics": metrics,
            }

    def _log_metrics(self, metrics: ModelMetrics):
        """Log model performance metrics"""
        logger.info("\nModel Performance:")
        logger.info(f"Training MSE: {metrics.train_mse:.4f}")
        logger.info(f"Test MSE: {metrics.test_mse:.4f}")
        logger.info(f"Training R²: {metrics.train_r2:.4f}")
        logger.info(f"Test R²: {metrics.test_r2:.4f}")
        logger.info(
            f"Features reduced from {metrics.n_features} to {metrics.n_components}"
        )

    def save_model(self, model_data: Dict, filename: str) -> None:
        """
        Save model, scaler, reducer, and metrics to disk
        Uses joblib for efficient saving of numpy arrays
        """
        import joblib
        from datetime import datetime

        save_path = self.output_dir / filename

        # Create a dictionary with all components and metadata
        save_dict = {
            "model": model_data["model"],
            "scaler": model_data["scaler"],
            "reducer": model_data["reducer"],
            "metrics": model_data["metrics"],
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "reduction_method": self.reduction_method,
                "n_components": self.n_components,
                "variance_threshold": self.variance_threshold,
            },
        }

        logger.info(f"Saving model to {save_path}")
        joblib.dump(save_dict, save_path)

    @staticmethod
    def load_model(model_path: Union[str, Path]) -> Dict:
        """
        Load saved model and associated components
        """
        import joblib

        model_path = Path(model_path)
        logger.info(f"Loading model from {model_path}")

        save_dict = joblib.load(model_path)

        logger.info("Model loaded successfully")
        logger.info(f"Model creation date: {save_dict['metadata']['creation_date']}")

        return save_dict

    def predict_batch(
        self, data: Union[pd.DataFrame, np.ndarray], batch_size: int = 1000
    ) -> np.ndarray:
        """Make predictions in memory-efficient batches with progress tracking"""
        if isinstance(data, pd.DataFrame):
            # Get the same columns as used in training
            logger.info("Preparing features for prediction...")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            embedding_cols = [col for col in numeric_cols if col not in ["y", "gdp"]]
            if len(embedding_cols) != 768:  # Expected number of features
                logger.warning(f"Expected 768 features, got {len(embedding_cols)}")
            data = data[embedding_cols].values

        n_samples = len(data)
        predictions = np.zeros(n_samples)

        for i in tqdm(range(0, n_samples, batch_size), desc="Predicting", unit="batch"):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx]

            # Scale batch
            batch_scaled = self.scaler.transform(batch)

            # Reduce dimensions
            if isinstance(self.reducer, MemoryEfficientPCA):
                batch_reduced = self.reducer.transform(batch_scaled)
            else:  # Feature selection
                batch_reduced = batch_scaled[:, self.reducer]

            # Predict
            predictions[i:end_idx] = self.model.predict(batch_reduced)

            # Clear memory
            gc.collect()

        return predictions

    def evaluate_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray, name: str = "Dataset"
    ) -> Dict[str, float]:
        """
        Evaluate predictions with multiple metrics
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            mean_absolute_percentage_error,
        )

        metrics = {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }

        logger.info(f"\nPerformance Metrics for {name}:")
        logger.info(f"MSE: {metrics['mse']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"MAE: {metrics['mae']:.4f}")
        logger.info(f"MAPE: {metrics['mape']:.4f}")
        logger.info(f"R²: {metrics['r2']:.4f}")

        return metrics

    def cross_validate(
        self,
        train_df: pd.DataFrame,
        n_splits: int = 5,
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation with memory efficiency and progress tracking"""
        from sklearn.model_selection import KFold

        logger.info(f"\nPerforming {n_splits}-fold cross-validation...")

        # Prepare data
        X, y, _, _ = self._prepare_data(train_df, train_df)

        # Initialize K-Fold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Initialize metrics storage
        metrics = {"mse": [], "rmse": [], "mae": [], "mape": [], "r2": []}

        # Perform CV with progress tracking
        for fold, (train_idx, val_idx) in enumerate(
            tqdm(kf.split(X), total=n_splits, desc="Cross-validation", unit="fold"), 1
        ):
            logger.info(f"\nProcessing fold {fold}/{n_splits}")

            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Scale features
            X_train_scaled, X_val_scaled, _ = self._scale_features(
                X_train_fold, X_val_fold
            )

            # Reduce dimensions
            X_train_reduced, X_val_reduced, _ = self._reduce_dimensions(
                X_train_scaled, X_val_scaled
            )

            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbose=0,  # Disable built-in verbosity in favor of tqdm
            )

            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                max_features=0.33,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train_reduced, y_train_fold)

            # Predict
            val_pred = model.predict(X_val_reduced)

            # Calculate metrics
            fold_metrics = self.evaluate_predictions(
                y_val_fold, val_pred, f"Fold {fold}"
            )

            # Store metrics
            for metric, value in fold_metrics.items():
                metrics[metric].append(value)

            # Clear memory
            gc.collect()

        # Calculate and log average metrics
        logger.info("\nCross-validation Results:")
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            logger.info(f"{metric.upper()}: {mean_value:.4f} (±{std_value:.4f})")

        return metrics

    def _plot_predictions(
        self,
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
    ):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_train, train_pred, alpha=0.5)
        plt.plot(
            [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2
        )
        plt.xlabel("Actual GDP")
        plt.ylabel("Predicted GDP")
        plt.title("Training Set: Actual vs Predicted")

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, test_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        plt.xlabel("Actual GDP")
        plt.ylabel("Predicted GDP")
        plt.title("Test Set: Actual vs Predicted")

        plt.tight_layout()
        plt.savefig(self.output_dir / "predictions.png", dpi=150)
        plt.close()

    def analyze_errors(
        self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.1
    ) -> Dict:
        """
        Analyze prediction errors to identify patterns and generate visualizations

        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Error threshold for significant errors

        Returns:
            Dictionary containing error analysis results
        """
        # Calculate errors
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        rel_errors = abs_errors / np.abs(y_true)

        # Identify significant errors
        significant_errors = rel_errors > threshold

        # Calculate error metrics
        error_metrics = {
            "mean_error": float(np.mean(errors)),
            "median_error": float(np.median(errors)),
            "std_error": float(np.std(errors)),
            "mean_abs_error": float(np.mean(abs_errors)),
            "median_abs_error": float(np.median(abs_errors)),
            "significant_errors_count": int(np.sum(significant_errors)),
            "significant_errors_percentage": float(np.mean(significant_errors) * 100),
        }

        # Generate visualizations if output directory is set
        if self.output_dir:
            self._plot_error_distribution(errors)
            self._plot_error_vs_true(y_true, errors)
            self._plot_qq(errors)

        # Log analysis results
        logger.info("\nError Analysis:")
        logger.info(f"Mean Error: {error_metrics['mean_error']:.4f}")
        logger.info(f"Median Error: {error_metrics['median_error']:.4f}")
        logger.info(f"Standard Deviation of Errors: {error_metrics['std_error']:.4f}")
        logger.info(f"Mean Absolute Error: {error_metrics['mean_abs_error']:.4f}")
        logger.info(
            f"Significant Errors: {error_metrics['significant_errors_count']} "
            f"({error_metrics['significant_errors_percentage']:.1f}%)"
        )

        return error_metrics

    def _plot_error_distribution(self, errors: np.ndarray) -> None:
        """Plot error distribution histogram"""
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=50, edgecolor="black")
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Error Distribution")
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_distribution.png", dpi=150)
        plt.close()

    def _plot_error_vs_true(self, y_true: np.ndarray, errors: np.ndarray) -> None:
        """Plot errors against true values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, errors, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("True Value")
        plt.ylabel("Error")
        plt.title("Error vs True Value")
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_vs_true.png", dpi=150)
        plt.close()

    def _plot_qq(self, errors: np.ndarray) -> None:
        """Plot Q-Q plot of errors"""
        from scipy import stats

        plt.figure(figsize=(8, 6))
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Errors")
        plt.tight_layout()
        plt.savefig(self.output_dir / "error_qq.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    # Example usage with memory optimization
    data_dir = Path("data")
    output_dir = data_dir / "model_outputs"

    # Load processed data
    train_df = pd.read_csv(data_dir / "processed" / "train_embeddings.csv")
    test_df = pd.read_csv(data_dir / "processed" / "test_embeddings.csv")

    # Initialize and train model with progress tracking
    trainer = ModelTrainer(
        reduction_method="pca",
        variance_threshold=0.90,  # Lower threshold since we're capping features
        max_features=45,  # Cap at 45 features
        output_dir=output_dir,
    )

    # Train model with memory optimization
    results = trainer.train(train_df, test_df)

    # Save the model
    trainer.save_model(results, "gdp_prediction_model.joblib")

    # Perform cross-validation with batching
    cv_results = trainer.cross_validate(
        train_df,
        n_splits=5,
        batch_size=1000,
    )

    # Analyze prediction errors
    with tqdm(total=2, desc="Analyzing predictions", unit="dataset") as pbar:
        train_pred = trainer.predict_batch(train_df, batch_size=1000)
        pbar.update(1)
        test_pred = trainer.predict_batch(test_df, batch_size=1000)
        pbar.update(1)

    trainer.analyze_errors(train_df["y"].values, train_pred)
    trainer.analyze_errors(test_df["y"].values, test_pred)

    logger.info("Model training and analysis complete!")
    logger.info(f"Results saved to {output_dir}")
