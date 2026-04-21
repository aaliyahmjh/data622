##########################################
# model.py - Train and evaluate salary prediction model
##########################################
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

from data622.paths import PROCESSED_DATA_DIR
from data622.features import make_preprocessor, get_model_columns


class SalaryPredictionModel:
    """Train and evaluate NYC payroll salary prediction model."""
    
    def __init__(self, model_type: str = "linear", alpha: float = None, regularization: str = None):
        """
        Initialize model.
        
        Args:
            model_type: "linear" (LinearRegression), "ridge" (Ridge), "lasso" (Lasso), "rf" (RandomForest)
            alpha: Regularization strength (for ridge/lasso)
            regularization: "ridge", "lasso", or "elasticnet" (for linear models)
        """
        self.model_type = model_type
        self.alpha = alpha
        self.regularization = regularization
        self.preprocessor = None
        self.model = None
        self.feature_cols = None
        self.cat_cols = None
        self.num_cols = None
        
    def load_data(self, train_path: str | Path = None, 
                  valid_path: str | Path = None,
                  test_path: str | Path = None):
        """Load train/valid/test CSV files."""
        if train_path is None:
            train_path = PROCESSED_DATA_DIR / "train_set.csv"
        if valid_path is None:
            valid_path = PROCESSED_DATA_DIR / "valid_set.csv"
        if test_path is None:
            test_path = PROCESSED_DATA_DIR / "test_set.csv"
            
        print("Loading data...")
        self.train_df = pd.read_csv(train_path)
        self.valid_df = pd.read_csv(valid_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✅ Train: {self.train_df.shape}")
        print(f"✅ Valid: {self.valid_df.shape}")
        print(f"✅ Test: {self.test_df.shape}")
        
    def prepare_features(self):
        """Get feature columns and create preprocessor."""
        print("\nPreparing features...")
        
        # Get categorical and numeric features
        self.cat_cols, self.num_cols = get_model_columns(self.train_df)
        self.feature_cols = self.cat_cols + self.num_cols
        
        print(f"Categorical features ({len(self.cat_cols)}): {self.cat_cols[:3]}...")
        print(f"Numeric features ({len(self.num_cols)}): {self.num_cols[:3]}...")
        
        # Create preprocessing pipeline
        self.preprocessor = make_preprocessor(self.train_df)
        
    def get_X_y(self, df: pd.DataFrame, target: str = "log_base_salary"):
        """Extract features and target from dataframe."""
        X = df[self.feature_cols].copy()
        y = df[target].copy() if target in df.columns else None
        return X, y
    
    def train(self, target: str = "log_base_salary"):
        """Train model on training data."""
        print(f"\nTraining {self.model_type} model (target: {target})...")
        
        # Get training features and target
        X_train, y_train = self.get_X_y(self.train_df, target)
        
        # Fit preprocessor and transform
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Initialize and train model
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "ridge":
            alpha = self.alpha if self.alpha is not None else 1.0
            self.model = Ridge(alpha=alpha)
        elif self.model_type == "lasso":
            alpha = self.alpha if self.alpha is not None else 0.001
            self.model = Lasso(alpha=alpha, max_iter=10000)
        elif self.model_type == "elasticnet":
            alpha = self.alpha if self.alpha is not None else 0.001
            self.model = ElasticNet(alpha=alpha, max_iter=10000)
        elif self.model_type == "rf":
            self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.fit(X_train_transformed, y_train)
        print(f"✅ Model trained successfully")
        
    def tune_xgboost(self, target: str = "log_base_salary", n_iter: int = 5, cv: int = 3):
        """Tune XGBoost hyperparameters using RandomizedSearchCV."""
        print(f"\n🔍 Tuning hyperparameters for XGBoost...")
        
        if self.model_type != "xgb":
            print("Model type must be 'xgb' to run this tuning function.")
            return

        X_train, y_train = self.get_X_y(self.train_df, target)
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        param_grid = {
            'n_estimators': [100, 200, 300], 
            'learning_rate': [0.01, 0.05, 0.1], 
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }

        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=n_iter, 
            cv=cv, scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
        
        search.fit(X_train_transformed, y_train)
        print(f"✅ Best parameters found: {search.best_params_}")
        
        self.model = search.best_estimator_
        print(f"✅ Model trained successfully")

    def cross_validate(self, target: str = "log_base_salary", cv: int = 5):
        """Perform cross-validation on validation set."""
        print(f"\nCross-validation on validation set (k={cv})...")
        
        X_valid, y_valid = self.get_X_y(self.valid_df, target)
        X_valid_transformed = self.preprocessor.transform(X_valid)
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_valid_transformed, y_valid, 
            cv=cv, scoring='r2', n_jobs=-1
        )
        
        print(f"CV R² scores: {cv_scores}")
        print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores
    
    def tune_regularization(self, target: str = "log_base_salary", cv: int = 5):
        """Tune regularization hyperparameter (alpha) for Ridge/Lasso."""
        if self.model_type not in ["ridge", "lasso", "elasticnet"]:
            print(f"⚠️  Tuning only works for ridge/lasso/elasticnet models, got {self.model_type}")
            return None
        
        print(f"\n🔍 Tuning {self.model_type.upper()} regularization (alpha)...")
        
        X_train, y_train = self.get_X_y(self.train_df, target)
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Define alpha values to test
        if self.model_type == "ridge":
            alphas = np.logspace(-3, 3, 20)  # Ridge: broader range
            model_class = Ridge
        elif self.model_type == "lasso":
            alphas = np.logspace(-5, 1, 20)  # Lasso: smaller alphas work better
            model_class = Lasso
        else:  # elasticnet
            alphas = np.logspace(-4, 1, 20)
            model_class = ElasticNet
        
        # GridSearchCV
        param_grid = {'alpha': alphas}
        base_model = model_class(max_iter=10000) if self.model_type != "ridge" else model_class()
        
        search = GridSearchCV(
            base_model, param_grid, cv=cv, 
            scoring='r2', n_jobs=-1, verbose=1
        )
        
        search.fit(X_train_transformed, y_train)
        
        best_alpha = search.best_params_['alpha']
        best_score = search.best_score_
        
        print(f"\n✅ Best alpha found: {best_alpha:.6f}")
        print(f"✅ Best CV R²: {best_score:.4f}")
        
        # Update model with best alpha
        self.alpha = best_alpha
        if self.model_type == "ridge":
            self.model = Ridge(alpha=best_alpha)
        elif self.model_type == "lasso":
            self.model = Lasso(alpha=best_alpha, max_iter=10000)
        else:
            self.model = ElasticNet(alpha=best_alpha, max_iter=10000)
        
        self.model.fit(X_train_transformed, y_train)
        
        return {
            'best_alpha': best_alpha,
            'best_cv_r2': best_score,
            'all_alphas': alphas,
            'cv_results': search.cv_results_
        }
        return cv_scores
    
    def evaluate(self, target: str = "log_base_salary"):
        """Evaluate model on test set."""
        print(f"\nEvaluating on test set...")
        
        X_test, y_test = self.get_X_y(self.test_df, target)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_transformed)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"R² Score:        {r2:.4f}")
        print(f"RMSE (log):      {rmse:.4f}")
        print(f"MAE (log):       {mae:.4f}")
        print(f"Sample size:     {len(y_test):,}")
        print(f"{'='*60}")
        
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "y_pred": y_pred,
            "y_test": y_test
        }
    
    def save_model(self, path: str | Path = None):
        """Save trained model and preprocessor."""
        if path is None:
            path = PROCESSED_DATA_DIR.parent.parent / "models" / f"salary_model_{self.model_type}.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols
        }, path)
        
        print(f"\n✅ Model saved to: {path}")
    
    def full_pipeline(self, model_type: str = "linear"):
        """Run complete training pipeline."""
        self.model_type = model_type
        self.load_data()
        self.prepare_features()
        self.train()
        self.cross_validate()
        results = self.evaluate()
        self.save_model()
        
        return results


def main():
    """Main training script."""
    print("\n" + "="*60)
    print("NYC PAYROLL SALARY PREDICTION MODEL")
    print("="*60)
    
    results_dict = {}
    
    # Train Linear Regression model
    print("\n🔵 Training LINEAR REGRESSION model...")
    model_lr = SalaryPredictionModel(model_type="linear")
    results_lr = model_lr.full_pipeline(model_type="linear")
    results_dict["Linear Regression"] = results_lr['r2']
    
    # Train Ridge Regression model with tuning
    print("\n\n🟣 Training RIDGE REGRESSION model (with hyperparameter tuning)...")
    model_ridge = SalaryPredictionModel(model_type="ridge")
    model_ridge.load_data()
    model_ridge.prepare_features()
    ridge_tune_results = model_ridge.tune_regularization(cv=5)
    model_ridge.cross_validate()
    results_ridge = model_ridge.evaluate()
    results_dict["Ridge Regression"] = results_ridge['r2']
    
    # Train Lasso Regression model with tuning
    print("\n\n🟡 Training LASSO REGRESSION model (with hyperparameter tuning)...")
    model_lasso = SalaryPredictionModel(model_type="lasso")
    model_lasso.load_data()
    model_lasso.prepare_features()
    lasso_tune_results = model_lasso.tune_regularization(cv=5)
    model_lasso.cross_validate()
    results_lasso = model_lasso.evaluate()
    results_dict["Lasso Regression"] = results_lasso['r2']
    
    # Train Random Forest model
    print("\n\n🟢 Training RANDOM FOREST model...")
    model_rf = SalaryPredictionModel(model_type="rf")
    results_rf = model_rf.full_pipeline(model_type="rf")
    results_dict["Random Forest"] = results_rf['r2']
    
    # Train XGBoost Model
    print("\n\n🔥 Training and Tuning XGBOOST model...")
    model_xgb = SalaryPredictionModel(model_type="xgb")
    model_xgb.load_data()
    model_xgb.prepare_features()
    model_xgb.tune_xgboost(n_iter=5, cv=3)  # Runs your hyperparameter tuning
    results_xgb = model_xgb.evaluate()
    model_xgb.save_model()

    # Compare results
    print("\n\n" + "="*60)
    print("MODEL COMPARISON - TEST SET RESULTS")
    print("="*60)
    
    # Sort by R² score
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model_name, r2_score) in enumerate(sorted_results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} {model_name:25s} R² = {r2_score:.4f}")
    
    print("="*60)
    print(f"Linear Regression R²: {results_lr['r2']:.4f}")
    print(f"Random Forest R²:     {results_rf['r2']:.4f}")
    print(f"Tuned XGBoost R²:     {results_xgb['r2']:.4f}")
    print(f"\n✅ Best model: {sorted_results[0][0]} with R² = {sorted_results[0][1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
