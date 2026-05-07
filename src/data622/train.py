##########################################
# train.py - Training on processed data 
##########################################

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import joblib



# Use relative imports
from .paths import MODELS_DIR, PROCESSED_DATA_DIR
from .features import fit_feature_stats, transform_with_stats, get_model_columns, make_preprocessor

MAX_TENURE_YEARS = 50  # add limit to tenure to prevent extreme outliers

def load_and_prepare_data():
    """
    Load train/valid/test splits and prepare them with feature engineering.
     Statistics computed only from training data.
    """
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load the pre-split data
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_set.csv")
    valid_df = pd.read_csv(PROCESSED_DATA_DIR / "valid_set.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_set.csv")
    
    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    
    # Fit statistics on TRAINING DATA ONLY
    print("\n" + "="*60)
    print("FITTING FEATURE STATISTICS (TRAINING DATA ONLY)")
    print("="*60)
    feature_stats = fit_feature_stats(train_df)
    
    # 🔒 Transform ALL datasets using training statistics
    print("\nTransforming training data...")
    train_df = transform_with_stats(train_df, feature_stats)
    
    print("Transforming validation data...")
    valid_df = transform_with_stats(valid_df, feature_stats)
    
    print("Transforming test data...")
    test_df = transform_with_stats(test_df, feature_stats)
    
    return train_df, valid_df, test_df, feature_stats


def prepare_features_and_target(df, target_col='log_base_salary'):
    """Extract features and target from dataframe."""
    # Get model columns
    cat_cols, num_cols = get_model_columns(df)
    feature_cols = cat_cols + num_cols
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    return X, y, feature_cols, cat_cols, num_cols


def train_and_evaluate_models(train_df, valid_df, test_df, feature_stats):
    """Train multiple models and evaluate on validation set."""
    
    # Prepare features for all sets
    X_train, y_train, feature_cols, cat_cols, num_cols = prepare_features_and_target(train_df)
    X_valid, y_valid, _, _, _ = prepare_features_and_target(valid_df)
    X_test, y_test, _, _, _ = prepare_features_and_target(test_df)
    
    # Create and fit preprocessor
    print("\n" + "="*60)
    print("CREATING PREPROCESSING PIPELINE")
    print("="*60)
    preprocessor = make_preprocessor(train_df)
    
    # Transform all sets
    print("Transforming features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_valid_transformed = preprocessor.transform(X_valid)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Dictionary to store results
    results = {}
    models = {}
    
    # ============================================
    # 1. Linear Regression
    # ============================================
    print("\n" + "="*60)
    print("TRAINING LINEAR REGRESSION")
    print("="*60)
    
    lr = LinearRegression()
    lr.fit(X_train_transformed, y_train)
    
    y_pred_valid = lr.predict(X_valid_transformed)
    y_pred_test = lr.predict(X_test_transformed)
    
    results['Linear Regression'] = {
        'valid_r2': r2_score(y_valid, y_pred_valid),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'valid_mae': mean_absolute_error(y_valid, y_pred_valid),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    models['Linear Regression'] = lr
    
    print(f"  Validation R²: {results['Linear Regression']['valid_r2']:.4f}")
    print(f"  Test R²:       {results['Linear Regression']['test_r2']:.4f}")
    
    # ============================================
    # 2. Ridge Regression
    # ============================================
    print("\n" + "="*60)
    print("TRAINING RIDGE REGRESSION")
    print("="*60)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_transformed, y_train)
    
    y_pred_valid = ridge.predict(X_valid_transformed)
    y_pred_test = ridge.predict(X_test_transformed)
    
    results['Ridge Regression'] = {
        'valid_r2': r2_score(y_valid, y_pred_valid),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'valid_mae': mean_absolute_error(y_valid, y_pred_valid),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    models['Ridge Regression'] = ridge
    
    print(f"  Validation R²: {results['Ridge Regression']['valid_r2']:.4f}")
    print(f"  Test R²:       {results['Ridge Regression']['test_r2']:.4f}")
    
    # ============================================
    # 3. Random Forest
    # ============================================
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf = RandomForestRegressor(
    n_estimators=50,        # Reduced from 100
    max_depth=10,           # Limit tree depth
    min_samples_split=50,   # Require more samples to split
    min_samples_leaf=25,    # Minimum samples per leaf
    random_state=42, 
    n_jobs=-1
)
    rf.fit(X_train_transformed, y_train)
    
    y_pred_valid = rf.predict(X_valid_transformed)
    y_pred_test = rf.predict(X_test_transformed)
    
    results['Random Forest'] = {
        'valid_r2': r2_score(y_valid, y_pred_valid),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'valid_mae': mean_absolute_error(y_valid, y_pred_valid),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    models['Random Forest'] = rf
    
    print(f"  Validation R²: {results['Random Forest']['valid_r2']:.4f}")
    print(f"  Test R²:       {results['Random Forest']['test_r2']:.4f}")
    
    # ============================================
    # 4. Gradient Boosting
    # ============================================
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING")
    print("="*60)
    
    gb = GradientBoostingRegressor(
    n_estimators=50,            # Reduced from 100
    learning_rate=0.1,
    max_depth=4,                # Shallower trees
    min_samples_split=100,      # Larger splits
    min_samples_leaf=50,
    subsample=0.7,
    random_state=42
    )
    gb.fit(X_train_transformed, y_train)
    
    y_pred_valid = gb.predict(X_valid_transformed)
    y_pred_test = gb.predict(X_test_transformed)
    
    results['Gradient Boosting'] = {
        'valid_r2': r2_score(y_valid, y_pred_valid),
        'valid_rmse': np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'valid_mae': mean_absolute_error(y_valid, y_pred_valid),
        'test_r2': r2_score(y_test, y_pred_test),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'test_mae': mean_absolute_error(y_test, y_pred_test)
    }
    models['Gradient Boosting'] = gb
    
    print(f"  Validation R²: {results['Gradient Boosting']['valid_r2']:.4f}")
    print(f"  Test R²:       {results['Gradient Boosting']['test_r2']:.4f}")
    
    return results, models, preprocessor, feature_cols, cat_cols, num_cols, feature_stats


def save_models(models, preprocessor, feature_cols, cat_cols, num_cols, feature_stats):
    """Save trained models and preprocessing objects."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        # Create filename (lowercase, replace spaces with underscores)
        filename = MODELS_DIR / f"salary_model_{model_name.lower().replace(' ', '_')}.pkl"
        
        # Save model with all necessary components
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'feature_cols': feature_cols,
            'cat_cols': cat_cols,
            'num_cols': num_cols,
            'feature_stats': feature_stats,
            'model_name': model_name
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Saved {model_name} to {filename}")


def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("MODEL COMPARISON - TEST SET RESULTS")
    print("="*60)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        model: {
            'R² Score': metrics['test_r2'],
            'RMSE (log)': metrics['test_rmse'],
            'MAE (log)': metrics['test_mae']
        }
        for model, metrics in results.items()
    }).T
    
    # Sort by R²
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    print(results_df.round(4))
    
    # Cross-validation scores for best model
    best_model_name = results_df.index[0]
    print(f"\n Best Model: {best_model_name} with R² = {results_df.iloc[0]['R² Score']:.4f}")
    
    # Also print actual dollar metrics if available
    print("\n" + "="*60)
    print("DOLLAR METRICS (for best model)")
    print("="*60)
    print("Note: Metrics are on log-transformed target")
    print("To get dollar metrics, exponentiate predictions")
    
    return results_df


def main():
    """Main training script."""
    print("\n" + "="*60)
    print("NYC PAYROLL SALARY PREDICTION MODEL")
    print("TRAINING PIPELINE")
    print("="*60)
    
    # Load and prepare data
    train_df, valid_df, test_df, feature_stats = load_and_prepare_data()
    
    # Train and evaluate models
    results, models, preprocessor, feature_cols, cat_cols, num_cols, feature_stats = \
        train_and_evaluate_models(train_df, valid_df, test_df, feature_stats)
    
    # Print results
    results_df = print_results(results)
    
    # Save models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    save_models(models, preprocessor, feature_cols, cat_cols, num_cols, feature_stats)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add parent directory to path for running directly
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    main()