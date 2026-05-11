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


class SalaryPredictor:
    """Load trained model and make predictions on new payroll data."""
    
    def __init__(self, model_path, feature_stats=None):
        self.model_path = Path(model_path)
        self.model_data = joblib.load(self.model_path)
        self.model = self.model_data['model']
        self.preprocessor = self.model_data['preprocessor']
        self.feature_cols = self.model_data['feature_cols']
        self.cat_cols = self.model_data['cat_cols']
        self.num_cols = self.model_data['num_cols']
        self.feature_stats = feature_stats or self.model_data.get('feature_stats')
        
        self.model_name = self.model_data.get('model_name', self.model_path.stem)
        self.model_r2 = self.model_data.get('test_r2', None)
        
        if self.feature_stats is None:
            print("⚠️  Warning: No feature statistics found in model. Predictions may have data leakage.")
    
    def predict(self, df, apply_feature_stats=True):
        X = df[self.feature_cols].copy()
        X_processed = self.preprocessor.transform(X)
        log_predictions = self.model.predict(X_processed)
        salary_predictions = np.exp(log_predictions)
        
        return salary_predictions, log_predictions
    
    def predict_with_confidence(self, df, actual_salary_col='base_salary'):
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
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'r2_score': self.model_r2,
            'feature_count': len(self.feature_cols) if self.feature_cols else 0
        }

def get_all_models_performance():
    models_performance = []
    model_files = list(MODELS_DIR.glob("salary_model_*.pkl"))
    
    for model_path in model_files:
        try:
            model_data = joblib.load(model_path)
            model_name = model_data.get('model_name', model_path.stem.replace('salary_model_', ''))
            
            r2_score = (
                model_data.get('test_r2') or 
                model_data.get('best_cv_r2') or 
                model_data.get('r2')
            )
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
    
    models_performance.sort(key=lambda x: x['r2_score'] if x['r2_score'] is not None else -np.inf, reverse=True)
    return models_performance

def get_best_model():
    models = get_all_models_performance()
    valid_models = [m for m in models if m['r2_score'] is not None]
    
    if valid_models:
        return valid_models[0]
    else:
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
    df = filter_model_population(df)
    df = add_tenure_proxy(df)
    return df

def quick_predict(model_name=None, test_year=None):
    if model_name:
        model_path = MODELS_DIR / model_name
        if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")
    else:
        best_model = get_best_model()
        if best_model:
            model_path = best_model['path']
            print(f"Using best model: {best_model['name']} (R² = {best_model['r2_score']:.4f})")
        else:
            raise FileNotFoundError("No models found in directory")
    
    print(f"Loading model: {model_path}...")
    predictor = SalaryPredictor(model_path)
    
    print("Loading and preparing test data...")
    df = load_salary_data()
    df = prepare_prediction_data(df)
    
    _, _, test_df = split_by_year(df)
    if test_year: test_df = test_df[test_df['fiscal_year'] == test_year]
    
    return predictor, test_df

def print_model_comparison():
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
    if models and models[0]['r2_score']:
        print(f"\n📌 Best Model: {models[0]['name']} (R² = {models[0]['r2_score']:.4f})")
    print("="*55 + "\n")

def main():
    print_model_comparison()
    try:
        predictor, test_df = quick_predict()
    except FileNotFoundError:
        print("❌ No models found. Please run train.py first.")
        return
    
    print(f"\n📊 Testing Predictions on {len(test_df)} samples\n")
    results = predictor.predict_with_confidence(test_df)
    
    print("Sample Predictions (first 10 rows):")
    display_cols = ['fiscal_year', 'base_salary', 'predicted_salary', 'salary_error_pct']
    available_cols = [c for c in display_cols if c in results.columns]
    if available_cols: print(results[available_cols].head(10))

# 🛡️ NYCSalaryStabilizer Logic embedded here for Streamlit
def predict_for_app(user_inputs: pd.DataFrame, model_name=None):
    if model_name:
        model_path = MODELS_DIR / model_name
    else:
        best_model = get_best_model()
        if best_model: model_path = best_model['path']
        else: raise FileNotFoundError("No models found")
    
    predictor = SalaryPredictor(model_path)
    user_inputs = user_inputs.copy()
    
    # 🛡️ GUARDRAIL 1: Tenure Normalization (Cap at 20)
    if 'tenure_years' in user_inputs.columns:
        user_inputs['tenure_years'] = user_inputs['tenure_years'].clip(upper=20)
    
    if 'regular_hours_worked' not in user_inputs.columns: user_inputs['regular_hours_worked'] = 40
    if 'pay_basis' not in user_inputs.columns: user_inputs['pay_basis'] = 'annual'
    
    salaries, log_salaries = predictor.predict(user_inputs)
    model_raw_dollars = salaries[0]
    
    # 🛡️ GUARDRAIL 2: Intercept Fix (70/30 Blend)
    historical_median = 0
    final_salary = model_raw_dollars
    
    if predictor.feature_stats is not None and 'title_avg_salary' in predictor.feature_stats:
        title = user_inputs['title_std'].iloc[0] if 'title_std' in user_inputs.columns else None
        if title and title in predictor.feature_stats['title_avg_salary']:
            historical_median = predictor.feature_stats['title_avg_salary'][title]
            # Blend: 70% Model, 30% Historical Median
            final_salary = (0.70 * model_raw_dollars) + (0.30 * historical_median)
            
    return {
        "expected_salary": round(final_salary, 2),
        "historical_median": round(historical_median, 2),
        "model_prediction_raw": round(model_raw_dollars, 2),
        "model_info": predictor.get_model_info()
    }

if __name__ == '__main__':
    main()