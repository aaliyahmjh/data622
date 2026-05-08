##########################################
# predict.py - Predicting on new data
# 🔒 LEAKAGE-SAFE VERSION - Reuses dataset.py functions
##########################################

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 🔒 FIXED: Use relative imports
from .paths import MODELS_DIR
from .dataset import load_salary_data, filter_model_population, add_tenure_proxy, split_by_year
from .features import transform_with_stats


class SalaryPredictor:
    """Load trained model and make predictions on new payroll data."""
    
    def __init__(self, model_path, feature_stats=None):
        """
        Load serialized model, preprocessor, and feature info.
        
        Args:
            model_path: Path to saved model
            feature_stats: Optional pre-loaded feature statistics
        """
        self.model_path = Path(model_path)
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        self.preprocessor = self.model_data['preprocessor']
        self.feature_cols = self.model_data['feature_cols']
        self.cat_cols = self.model_data['cat_cols']
        self.num_cols = self.model_data['num_cols']
        self.feature_stats = feature_stats or self.model_data.get('feature_stats')
        
        # Extract model performance if saved
        self.model_name = self.model_data.get('model_name', self.model_path.stem)
        self.model_r2 = self.model_data.get('test_r2', None)
        
        if self.feature_stats is None:
            print("⚠️  Warning: No feature statistics found in model. Predictions may have data leakage.")
    
    def predict(self, df, apply_feature_stats=True):
        """
        Predict salaries on new data.
        
        Args:
            df: Raw dataframe (before feature engineering)
            apply_feature_stats: Whether to apply training statistics (recommended=True)
        """
        # 🔒 Apply feature statistics if available (prevents leakage)
        if apply_feature_stats and self.feature_stats is not None:
            df = transform_with_stats(df, self.feature_stats)
        
        # Use only the required feature columns
        X = df[self.feature_cols].copy()
        
        # Preprocess features
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions (log scale)
        log_predictions = self.model.predict(X_processed)
        
        # Transform back to original salary scale
        salary_predictions = np.exp(log_predictions)
        
        return salary_predictions, log_predictions
    
    def predict_with_confidence(self, df, actual_salary_col='base_salary'):
        """
        Predict with confidence metrics if actual values available.
        
        Args:
            df: Raw dataframe with actual salaries
            actual_salary_col: Column name containing actual salaries
        """
        salaries, log_salaries = self.predict(df)
        
        result = df.copy()
        result['predicted_salary'] = salaries
        result['predicted_log_salary'] = log_salaries
        
        if actual_salary_col in df.columns:
            result['actual_salary'] = df[actual_salary_col]
            result['actual_log_salary'] = np.log(df[actual_salary_col])
            result['log_error'] = result['predicted_log_salary'] - result['actual_log_salary']
            result['salary_error_pct'] = (result['predicted_salary'] - df[actual_salary_col]) / df[actual_salary_col] * 100
        
        return result
    
    def get_model_info(self):
        """Return model metadata."""
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'r2_score': self.model_r2,
            'feature_count': len(self.feature_cols) if self.feature_cols else 0
        }

        # Group Rare Titles
        # Look at the historical count from the reference table
        job_count = input_df['count_of_job_titles'].values[0]
        input_df['title_std_grouped'] = input_df['title_std'] if job_count >= 100 else 'other_title'

def get_all_models_performance():
    """
    Scan models directory and extract performance metrics from saved models.
    Returns sorted list of models by R² score.
    """
    models_performance = []
    
    # Find all model files
    model_files = list(MODELS_DIR.glob("salary_model_*.pkl"))
    
    for model_path in model_files:
        try:
            # Load model data
            model_data = joblib.load(model_path)
            
            # Extract model name and R² score
            model_name = model_data.get('model_name', model_path.stem.replace('salary_model_', ''))
            
            # Try different possible keys for R² score
            r2_score = (
                model_data.get('test_r2') or 
                model_data.get('best_cv_r2') or 
                model_data.get('r2')
            )
            
            # Also check if model has cross-validation scores
            cv_scores = model_data.get('cv_scores', None)
            
            models_performance.append({
                'name': model_name,
                'r2_score': r2_score,
                'path': model_path,
                'cv_scores': cv_scores,
                'has_stats': model_data.get('feature_stats') is not None
            })
            
        except Exception as e:
            print(f"⚠️  Could not load {model_path.name}: {e}")
            continue
    
    # Sort by R² score (highest first)
    models_performance.sort(key=lambda x: x['r2_score'] if x['r2_score'] is not None else -np.inf, reverse=True)
    
    return models_performance


def get_best_model():
    """Return the best performing model based on saved metrics."""
    models = get_all_models_performance()
    
    # Filter to models with valid R² scores
    valid_models = [m for m in models if m['r2_score'] is not None]
    
    if valid_models:
        return valid_models[0]
    else:
        # Fallback to linear regression if no metrics found
        default_path = MODELS_DIR / "salary_model_linear.pkl"
        if default_path.exists():
            return {
                'name': 'linear',
                'r2_score': None,
                'path': default_path,
                'cv_scores': None,
                'has_stats': False
            }
        return None


def prepare_prediction_data(df):
    """
    Prepare data for prediction using existing dataset functions.
    🔒 NO CODE REPETITION - Reuses functions from dataset.py
    """
    # Apply all the standard data preparation steps
    df = filter_model_population(df)
    df = add_tenure_proxy(df)
    return df


def quick_predict(model_name=None, test_year=None):
    """
    Quick prediction function for testing.
    
    Args:
        model_name: Name of model file in MODELS_DIR (None = use best model)
        test_year: Specific year to test (default: uses split_by_year)
    """
    # Determine which model to use
    if model_name:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        best_model = get_best_model()
        if best_model:
            model_path = best_model['path']
            print(f"Using best model: {best_model['name']} (R² = {best_model['r2_score']:.4f})")
        else:
            raise FileNotFoundError("No models found in directory")
    
    # Load model
    print(f"Loading model: {model_path}...")
    predictor = SalaryPredictor(model_path)
    
    # Load and prepare test data using existing functions
    print("Loading and preparing test data...")
    df = load_salary_data()
    df = prepare_prediction_data(df)
    
    # Split and get test set
    _, _, test_df = split_by_year(df)
    
    if test_year:
        test_df = test_df[test_df['fiscal_year'] == test_year]
    
    return predictor, test_df


def print_model_comparison():
    """Print comparison of all available models."""
    models = get_all_models_performance()
    
    if not models:
        print("❌ No models found in", MODELS_DIR)
        return
    
    print("\n" + "="*55)
    print("MODEL COMPARISON (by R² Score)")
    print("="*55)
    
    for i, model in enumerate(models, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        r2_str = f"{model['r2_score']:.4f}" if model['r2_score'] is not None else "N/A"
        leakage_status = "✅" if model['has_stats'] else "⚠️"
        print(f"{emoji} {model['name']:25s} R² = {r2_str:>8}  {leakage_status}")
    
    print("="*55)
    print("✅ = Leakage-safe | ⚠️ = Check feature_stats")
    
    if models and models[0]['r2_score']:
        print(f"\n📌 Best Model: {models[0]['name']} (R² = {models[0]['r2_score']:.4f})")
    
    print("="*55 + "\n")


def main():
    """Example: Load model and predict on test data."""
    # Dynamic model comparison
    print_model_comparison()
    
    # Use best model automatically
    try:
        predictor, test_df = quick_predict()  # Automatically selects best model
    except FileNotFoundError:
        print("❌ No models found. Please run train.py first.")
        return
    
    # Make predictions
    print(f"\n📊 Testing Predictions on {len(test_df)} samples\n")
    results = predictor.predict_with_confidence(test_df)
    
    print("Sample Predictions (first 10 rows):")
    display_cols = ['fiscal_year', 'base_salary', 'predicted_salary', 'salary_error_pct']
    available_cols = [c for c in display_cols if c in results.columns]
    if available_cols:
        print(results[available_cols].head(10))
    
    # Calculate metrics
    if 'salary_error_pct' in results.columns:
        print(f"\n📈 Prediction Accuracy Metrics:")
        print(f"  Mean Absolute Error:     {results['salary_error_pct'].abs().mean():.2f}%")
        print(f"  Median Error:            {results['salary_error_pct'].median():.2f}%")
        print(f"  Std Dev:                 {results['salary_error_pct'].std():.2f}%")
        print(f"  Min Error:               {results['salary_error_pct'].min():.2f}%")
        print(f"  Max Error:               {results['salary_error_pct'].max():.2f}%")
    
    # Show model info
    model_info = predictor.get_model_info()
    print(f"\n📌 Model Info:")
    print(f"  Model: {model_info['model_name']}")
    if model_info['r2_score']:
        print(f"  R² Score: {model_info['r2_score']:.4f}")
    print(f"  Features: {model_info['feature_count']}")
    print(f"  Path: {model_info['model_path']}")


# 🔒 NEW: Function for predicting on custom input data (for your app)
def predict_for_app(user_inputs: pd.DataFrame, model_name=None):
    """
    Make predictions for Streamlit app.
    
    Args:
        user_inputs: DataFrame with columns: 
                     ['title_std', 'agency_std', 'tenure_years', 'fiscal_year', 'pay_basis', 'regular_hours_worked']
        model_name: Name of model file (None = use best model)
    """
    # Determine which model to use
    if model_name:
        model_path = MODELS_DIR / model_name
    else:
        best_model = get_best_model()
        if best_model:
            model_path = best_model['path']
        else:
            raise FileNotFoundError("No models found")
    
    # Load model
    predictor = SalaryPredictor(model_path)
    
    # Prepare the input data (add required columns with defaults)
    user_inputs = user_inputs.copy()
    
    # Add required columns with default values if missing
    if 'regular_hours_worked' not in user_inputs.columns:
        user_inputs['regular_hours_worked'] = 40
    
    if 'pay_basis' not in user_inputs.columns:
        user_inputs['pay_basis'] = 'annual'
    
    # Make prediction
    salaries, log_salaries = predictor.predict(user_inputs)
    
    # Get model info
    model_info = predictor.get_model_info()
    
    return salaries, log_salaries, model_info

        return {
            "expected_salary": round(final_salary, 2),
            "historical_median": round(historical_median, 2),
            "model_prediction_raw": round(model_raw_dollars, 2)
        }

if __name__ == '__main__':
    main()
