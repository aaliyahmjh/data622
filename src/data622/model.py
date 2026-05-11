##########################################
# model.py - Train and evaluate salary prediction model
##########################################
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shutil

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# 🔒 FIXED: Use relative imports
from .paths import PROCESSED_DATA_DIR
from .features import make_preprocessor, get_model_columns


class SalaryPredictionModel:
    """Train and evaluate NYC payroll salary prediction model."""
    
    def __init__(self, model_type: str = "linear", alpha: float = None, regularization: str = None):
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
        self.train_df = pd.read_csv(train_path, low_memory=False)
        self.valid_df = pd.read_csv(valid_path, low_memory=False)
        self.test_df = pd.read_csv(test_path, low_memory=False)
        
        print(f"✅ Train: {self.train_df.shape}")
        print(f"✅ Valid: {self.valid_df.shape}")
        print(f"✅ Test: {self.test_df.shape}")
        
    def prepare_features(self):
        """Get feature columns and create preprocessor."""
        print("\nPreparing features...")
        
        self.cat_cols, self.num_cols = get_model_columns(self.train_df)
        self.feature_cols = self.cat_cols + self.num_cols
        
        if len(self.feature_cols) == 0:
            print("⚠️  Warning: No features found! Using fallback features.")
            self.cat_cols = ["agency_std"] if "agency_std" in self.train_df.columns else []
            self.num_cols = ["fiscal_year", "tenure_years", "log_base_salary"]
            self.num_cols = [c for c in self.num_cols if c in self.train_df.columns]
            self.feature_cols = self.cat_cols + self.num_cols
        
        print(f"Categorical features ({len(self.cat_cols)}): {self.cat_cols[:3] if self.cat_cols else 'None'}...")
        print(f"Numeric features ({len(self.num_cols)}): {self.num_cols[:3] if self.num_cols else 'None'}...")
        
        try:
            self.preprocessor = make_preprocessor(self.train_df)
        except Exception as e:
            print(f"⚠️  Error creating preprocessor: {e}")
            from sklearn.preprocessing import StandardScaler
            self.preprocessor = StandardScaler()
        
    def get_X_y(self, df: pd.DataFrame, target: str = "log_base_salary"):
        """Extract features and target from dataframe."""
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
        
        X_train, y_train = self.get_X_y(self.train_df, target)
        
        if X_train.empty or y_train is None:
            print("❌ No features or target found for training!")
            return
        
        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
        except Exception as e:
            print(f"⚠️  Error in preprocessor: {e}")
            X_train_transformed = X_train.values if hasattr(X_train, 'values') else X_train
        
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
        """Tune XGBoost hyperparameters."""
        print(f"\n🔍 Tuning hyperparameters for XGBoost...")
        if self.model_type != "xgb":
            return

        X_train, y_train = self.get_X_y(self.train_df, target)
        if X_train.empty or y_train is None: return
            
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
        
        if X_valid.empty or y_valid is None: return None
            
        try:
            X_valid_transformed = self.preprocessor.transform(X_valid)
        except:
            X_valid_transformed = X_valid.values if hasattr(X_valid, 'values') else X_valid
        
        cv_scores = cross_val_score(self.model, X_valid_transformed, y_valid, cv=cv, scoring='r2', n_jobs=-1)
        print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return cv_scores
    
    def tune_regularization(self, target: str = "log_base_salary", cv: int = 5):
        """Tune regularization hyperparameter (alpha) for Ridge/Lasso."""
        if self.model_type not in ["ridge", "lasso", "elasticnet"]: return None
        print(f"\n🔍 Tuning {self.model_type.upper()} regularization (alpha)...")
        
        X_train, y_train = self.get_X_y(self.train_df, target)
        if X_train.empty or y_train is None: return None
            
        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
        except:
            X_train_transformed = X_train.values if hasattr(X_train, 'values') else X_train
        
        if self.model_type == "ridge":
            alphas, model_class = np.logspace(-3, 3, 10), Ridge
        elif self.model_type == "lasso":
            alphas, model_class = np.logspace(-5, 1, 10), Lasso
        else:
            alphas, model_class = np.logspace(-4, 1, 10), ElasticNet
        
        param_grid = {'alpha': alphas}
        base_model = model_class(max_iter=10000) if self.model_type in ["lasso", "elasticnet"] else model_class()
        
        search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1)
        search.fit(X_train_transformed, y_train)
        
        best_alpha = search.best_params_['alpha']
        self.alpha = best_alpha
        
        if self.model_type == "ridge": self.model = Ridge(alpha=best_alpha)
        elif self.model_type == "lasso": self.model = Lasso(alpha=best_alpha, max_iter=10000)
        else: self.model = ElasticNet(alpha=best_alpha, max_iter=10000)
        
        self.model.fit(X_train_transformed, y_train)
        return {'best_alpha': best_alpha, 'best_cv_r2': search.best_score_}
    
    def evaluate(self, target: str = "log_base_salary"):
        """Evaluate model on test set."""
        print(f"\nEvaluating on test set...")
        X_test, y_test = self.get_X_y(self.test_df, target)
        
        if X_test.empty or y_test is None: return {"r2": 0, "rmse": 0, "mae": 0}
            
        try:
            X_test_transformed = self.preprocessor.transform(X_test)
        except:
            X_test_transformed = X_test.values if hasattr(X_test, 'values') else X_test
        
        y_pred = self.model.predict(X_test_transformed)
        r2 = r2_score(y_test, y_pred)
        
        print(f"R² Score:        {r2:.4f}")
        return {"r2": r2, "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), "mae": mean_absolute_error(y_test, y_pred)}
    
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
            "num_cols": self.num_cols,
            "model_type": self.model_type
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
    print("="*60)
    
    results_dict = {}
    
    # 1. Linear Regression
    print("\n🔵 Training LINEAR REGRESSION model...")
    try:
        model_lr = SalaryPredictionModel(model_type="linear")
        results_lr = model_lr.full_pipeline(model_type="linear")
        results_dict["Linear Regression"] = results_lr['r2']
    except Exception as e: print(f"❌ Linear Regression failed: {e}")
    
    # 2. Ridge Regression
    print("\n\n🟣 Training RIDGE REGRESSION model...")
    try:
        model_ridge = SalaryPredictionModel(model_type="ridge")
        model_ridge.load_data()
        model_ridge.prepare_features()
        model_ridge.tune_regularization(cv=3)
        model_ridge.cross_validate()
        results_dict["Ridge Regression"] = model_ridge.evaluate()['r2']
        model_ridge.save_model()
    except Exception as e: print(f"❌ Ridge Regression failed: {e}")

    # 3. Lasso Regression 
    print("\n\n🟡 Training LASSO REGRESSION model...")
    try:
        model_lasso = SalaryPredictionModel(model_type="lasso")
        model_lasso.load_data()
        model_lasso.prepare_features()
        model_lasso.tune_regularization(cv=3)
        model_lasso.cross_validate()
        results_dict["Lasso Regression"] = model_lasso.evaluate()['r2']
        model_lasso.save_model()
    except Exception as e: print(f"❌ Lasso Regression failed: {e}")
    
    # 4. Random Forest
    print("\n\n🟢 Training RANDOM FOREST model...")
    try:
        model_rf = SalaryPredictionModel(model_type="rf")
        results_rf = model_rf.full_pipeline(model_type="rf")
        results_dict["Random Forest"] = results_rf['r2']
    except Exception as e: print(f"❌ Random Forest failed: {e}")

    # 5. XGBoost 
    print("\n\n🔥 Training XGBOOST model...")
    try:
        model_xgb = SalaryPredictionModel(model_type="xgb")
        model_xgb.load_data()
        model_xgb.prepare_features()
        model_xgb.tune_xgboost(n_iter=5, cv=3)
        model_xgb.cross_validate()
        results_dict["XGBoost"] = model_xgb.evaluate()['r2']
        model_xgb.save_model()
    except Exception as e: print(f"❌ XGBoost failed: {e}")

    # Compare results and Save Best Model
    if results_dict:
        print("\n\n" + "="*60)
        print("MODEL COMPARISON - TEST SET RESULTS")
        print("="*60)
        
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, r2_score) in enumerate(sorted_results, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"{emoji} {model_name:25s} R² = {r2_score:.4f}")
        
        best_model_name = sorted_results[0][0]
        print("="*60)
        print(f"\n✅ Best model: {best_model_name} with R² = {sorted_results[0][1]:.4f}")
        
        # Crowning logic
        file_map = {
            "Linear Regression": "salary_model_linear.pkl",
            "Ridge Regression": "salary_model_ridge.pkl",
            "Lasso Regression": "salary_model_lasso.pkl",
            "Random Forest": "salary_model_rf.pkl",
            "XGBoost": "salary_model_xgb.pkl"
        }
        
        best_file = file_map.get(best_model_name)
        if best_file:
            models_dir = Path("models")
            source_path = models_dir / best_file
            dest_path = models_dir / "salary_model_best.pkl"
            if source_path.exists():
                shutil.copy(source_path, dest_path)
                print(f"🏆 Champion saved! Copied {best_file} to salary_model_best.pkl")
        print("="*60)

if __name__ == "__main__":
    main()