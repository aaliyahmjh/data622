##########################################
# model.py - Train and evaluate salary prediction model
##########################################
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# 🔒 FIXED: Use relative imports
from .paths import PROCESSED_DATA_DIR
from .features import (make_preprocessor, get_model_columns, 
                       fit_feature_stats, transform_with_stats)


class SalaryPredictionModel:
    """Train and evaluate NYC payroll salary prediction model."""
    
    def __init__(self, model_type: str = "linear", alpha: float = None, regularization: str = None):
        """
        Initialize model.
        
        Args:
            model_type: "linear", "ridge", "lasso", "rf", "xgb"
            alpha: Regularization strength (for ridge/lasso)
            regularization: "ridge", "lasso", or "elasticnet"
        """
        self.model_type = model_type
        self.alpha = alpha
        self.regularization = regularization
        self.preprocessor = None
        self.model = None
        self.feature_cols = None
        self.cat_cols = None
        self.num_cols = None
        self.feature_stats = None
        
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
        
        # 🔒 Check if required columns exist before fitting stats
        required_for_stats = ["title_std", "agency_std", "base_salary"]
        missing_cols = [c for c in required_for_stats if c not in self.train_df.columns]
        
        if missing_cols:
            print(f"⚠️  Warning: Missing columns for feature stats: {missing_cols}")
            print("Creating placeholder columns...")
            for col in missing_cols:
                if col == "title_std":
                    self.train_df["title_std"] = "unknown"
                    self.valid_df["title_std"] = "unknown"
                    self.test_df["title_std"] = "unknown"
                elif col == "agency_std":
                    self.train_df["agency_std"] = "unknown"
                    self.valid_df["agency_std"] = "unknown"
                    self.test_df["agency_std"] = "unknown"
                elif col == "base_salary":
                    # This should already exist from dataset.py
                    pass
        
        # 🔒 Fit feature statistics on training data only
        print("Fitting feature statistics on training data...")
        try:
            self.feature_stats = fit_feature_stats(self.train_df)
        except Exception as e:
            print(f"⚠️  Could not fit feature stats: {e}")
            print("Creating empty stats...")
            self.feature_stats = {
                'title_counts': pd.Series(dtype='int64'),
                'agency_counts': pd.Series(dtype='int64'),
                'title_avg_salary': pd.Series(dtype='float64')
            }
        
        # 🔒 Transform all datasets using training statistics
        print("Transforming data with training statistics...")
        try:
            self.train_df = transform_with_stats(self.train_df, self.feature_stats)
            self.valid_df = transform_with_stats(self.valid_df, self.feature_stats)
            self.test_df = transform_with_stats(self.test_df, self.feature_stats)
        except Exception as e:
            print(f"⚠️  Error in transform_with_stats: {e}")
            print("Applying fallback transformations...")
            # Apply minimal transformations
            if "base_salary" in self.train_df.columns:
                self.train_df["log_base_salary"] = np.log(self.train_df["base_salary"].clip(lower=1))
                self.valid_df["log_base_salary"] = np.log(self.valid_df["base_salary"].clip(lower=1))
                self.test_df["log_base_salary"] = np.log(self.test_df["base_salary"].clip(lower=1))
        
        print(f"✅ Train: {self.train_df.shape}")
        print(f"✅ Valid: {self.valid_df.shape}")
        print(f"✅ Test: {self.test_df.shape}")
        
    def prepare_features(self):
        """Get feature columns and create preprocessor."""
        print("\nPreparing features...")
        
        # Get categorical and numeric features
        self.cat_cols, self.num_cols = get_model_columns(self.train_df)
        self.feature_cols = self.cat_cols + self.num_cols
        
        # 🔒 Check if we have any features
        if len(self.feature_cols) == 0:
            print("⚠️  Warning: No features found! Using fallback features.")
            # Use basic fallback features
            self.cat_cols = ["agency_std"] if "agency_std" in self.train_df.columns else []
            self.num_cols = ["fiscal_year", "tenure_years", "log_base_salary"]
            self.num_cols = [c for c in self.num_cols if c in self.train_df.columns]
            self.feature_cols = self.cat_cols + self.num_cols
        
        print(f"Categorical features ({len(self.cat_cols)}): {self.cat_cols[:3] if self.cat_cols else 'None'}...")
        print(f"Numeric features ({len(self.num_cols)}): {self.num_cols[:3] if self.num_cols else 'None'}...")
        
        # Create preprocessing pipeline
        try:
            self.preprocessor = make_preprocessor(self.train_df)
        except Exception as e:
            print(f"⚠️  Error creating preprocessor: {e}")
            # Create a simple fallback preprocessor
            from sklearn.preprocessing import StandardScaler
            self.preprocessor = StandardScaler()
        
    def get_X_y(self, df: pd.DataFrame, target: str = "log_base_salary"):
        """Extract features and target from dataframe."""
        # 🔒 Check if feature columns exist
        available_features = [c for c in self.feature_cols if c in df.columns]
        if len(available_features) != len(self.feature_cols):
            missing = set(self.feature_cols) - set(available_features)
            print(f"⚠️  Warning: Missing features: {missing}")
        
        X = df[available_features].copy() if available_features else pd.DataFrame()
        y = df[target].copy() if target in df.columns else None
        return X, y
    
    def train(self, target: str = "log_base_salary"):
        """Train model on training data."""
        print(f"\nTraining {self.model_type} model (target: {target})...")
        
        # Get training features and target
        X_train, y_train = self.get_X_y(self.train_df, target)
        
        # 🔒 Check if we have data to train on
        if X_train.empty or y_train is None:
            print("❌ No features or target found for training!")
            return
        
        # Fit preprocessor and transform
        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
        except Exception as e:
            print(f"⚠️  Error in preprocessor: {e}")
            # Fallback: use raw features
            X_train_transformed = X_train.values if hasattr(X_train, 'values') else X_train
        
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
            self.model = RandomForestRegressor(
                n_estimators=100, 
                n_jobs=-1, 
                random_state=42,
                max_depth=10,
                min_samples_split=50
            )
        elif self.model_type == "xgb":
            self.model = XGBRegressor(random_state=42, n_jobs=-1)
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
        
        if X_train.empty or y_train is None:
            print("❌ No data for tuning!")
            return
            
        try:
            X_train_transformed = self.preprocessor.transform(X_train)
        except:
            X_train_transformed = X_train.values if hasattr(X_train, 'values') else X_train
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        
        param_grid = {
            'n_estimators': [100, 200], 
            'learning_rate': [0.05, 0.1], 
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }

        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=min(n_iter, 4), 
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
        
        if X_valid.empty or y_valid is None:
            print("⚠️  No validation data for cross-validation")
            return None
            
        try:
            X_valid_transformed = self.preprocessor.transform(X_valid)
        except:
            X_valid_transformed = X_valid.values if hasattr(X_valid, 'values') else X_valid
        
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
        
        if X_train.empty or y_train is None:
            print("❌ No data for tuning!")
            return None
            
        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
        except:
            X_train_transformed = X_train.values if hasattr(X_train, 'values') else X_train
        
        # Define alpha values to test
        if self.model_type == "ridge":
            alphas = np.logspace(-3, 3, 10)  # Reduced from 20 for speed
            model_class = Ridge
        elif self.model_type == "lasso":
            alphas = np.logspace(-5, 1, 10)
            model_class = Lasso
        else:  # elasticnet
            alphas = np.logspace(-4, 1, 10)
            model_class = ElasticNet
        
        # GridSearchCV
        param_grid = {'alpha': alphas}
        base_model = model_class(max_iter=10000) if self.model_type in ["lasso", "elasticnet"] else model_class()
        
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
    
    def evaluate(self, target: str = "log_base_salary"):
        """Evaluate model on test set."""
        print(f"\nEvaluating on test set...")
        
        X_test, y_test = self.get_X_y(self.test_df, target)
        
        if X_test.empty or y_test is None:
            print("❌ No test data for evaluation!")
            return {"r2": 0, "rmse": 0, "mae": 0, "y_pred": None, "y_test": None}
            
        try:
            X_test_transformed = self.preprocessor.transform(X_test)
        except:
            X_test_transformed = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Make predictions
        y_pred = self.model.predict(X_test_transformed)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS - {self.model_type.upper()}")
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
        """Save trained model, preprocessor, and feature statistics."""
        if path is None:
            path = PROCESSED_DATA_DIR.parent.parent / "models" / f"salary_model_{self.model_type}.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            "model": self.model,
            "preprocessor": self.preprocessor,
            "feature_cols": self.feature_cols,
            "cat_cols": self.cat_cols,
            "num_cols": self.num_cols,
            "feature_stats": self.feature_stats,
            "model_type": self.model_type
        }, path)
        
        print(f"\n✅ Model saved to: {path}")
    
    def load_model(self, path: str | Path):
        """Load trained model with all components."""
        path = Path(path)
        data = joblib.load(path)
        
        self.model = data["model"]
        self.preprocessor = data["preprocessor"]
        self.feature_cols = data["feature_cols"]
        self.cat_cols = data["cat_cols"]
        self.num_cols = data["num_cols"]
        self.feature_stats = data.get("feature_stats")
        self.model_type = data.get("model_type", "unknown")
        
        print(f"\n✅ Model loaded from: {path}")
    
    def predict(self, df: pd.DataFrame):
        """Make predictions on new data."""
        if self.feature_stats is None:
            raise ValueError("Feature statistics not found. Load model with load_model() first.")
        
        df_transformed = transform_with_stats(df, self.feature_stats)
        X = df_transformed[self.feature_cols]
        X_transformed = self.preprocessor.transform(X)
        
        predictions = self.model.predict(X_transformed)
        return predictions
    
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
    print("="*60)
    
    results_dict = {}
    
    # Train Linear Regression model
    print("\n🔵 Training LINEAR REGRESSION model...")
    try:
        model_lr = SalaryPredictionModel(model_type="linear")
        results_lr = model_lr.full_pipeline(model_type="linear")
        results_dict["Linear Regression"] = results_lr['r2']
    except Exception as e:
        print(f"❌ Linear Regression failed: {e}")
    
    # Train Ridge Regression model
    print("\n\n🟣 Training RIDGE REGRESSION model...")
    try:
        model_ridge = SalaryPredictionModel(model_type="ridge")
        model_ridge.load_data()
        model_ridge.prepare_features()
        model_ridge.tune_regularization(cv=3)
        model_ridge.cross_validate()
        results_ridge = model_ridge.evaluate()
        results_dict["Ridge Regression"] = results_ridge['r2']
        model_ridge.save_model()
    except Exception as e:
        print(f"❌ Ridge Regression failed: {e}")
    
    # Train Random Forest model
    print("\n\n🟢 Training RANDOM FOREST model...")
    try:
        model_rf = SalaryPredictionModel(model_type="rf")
        results_rf = model_rf.full_pipeline(model_type="rf")
        results_dict["Random Forest"] = results_rf['r2']
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    # Compare results
    if results_dict:
        print("\n\n" + "="*60)
        print("MODEL COMPARISON - TEST SET RESULTS")
        print("="*60)
        
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, r2_score) in enumerate(sorted_results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji} {model_name:25s} R² = {r2_score:.4f}")
        
        print("="*60)
        print(f"\n✅ Best model: {sorted_results[0][0]} with R² = {sorted_results[0][1]:.4f}")
        print("="*60)
    else:
        print("\n❌ No models trained successfully!")


if __name__ == "__main__":
    main()