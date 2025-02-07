from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

def fit_lasso_time_series(df, target_var, independent_vars, alpha=0.01, n_splits=5):
    """
    Fits a LASSO regression model with time-series aware cross-validation, avoiding data leakage.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        target_var (str): The name of the dependent variable (target).
        independent_vars (list of str): List of independent variable names (features).
        alpha (float): Regularization strength (default=0.01).
        n_splits (int): Number of splits for time series cross-validation.
    
    Returns:
        list: List of selected features with non-zero coefficients.
    """
    X = df[independent_vars].values
    y = df[target_var].values

    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Prepare to store LASSO results
    best_lasso = None
    best_score = -np.inf

    for train_idx, test_idx in tscv.split(X):
        # Split train and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale using training set only (avoiding leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Use same scaler for test set

        # Fit LASSO on the training data (no CV, just a fixed alpha)
        lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train_scaled, y_train)

        # Evaluate model and store the best LASSO (optional)
        score = lasso.score(X_test_scaled, y_test)
        if score > best_score:
            best_score = score
            best_lasso = lasso

    # Extract selected features (non-zero coefficients from the best model)
    selected_features = np.array(independent_vars)[best_lasso.coef_ != 0].tolist()

    return selected_features

# Example usage:
# selected_vars = fit_lasso_time_series(df, 'y', candidate_features, alpha=0.01)
# selected_vars
