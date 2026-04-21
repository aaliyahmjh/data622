##########################################
# predict.py - Predicting on new data
##########################################

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from data622.paths import MODELS_DIR


class SalaryPredictor:
    """Load trained model and make predictions on new payroll data."""
    
    def __init__(self, model_path):
        """Load serialized model, preprocessor, and feature info."""
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.preprocessor = self.model_data['preprocessor']
        self.feature_cols = self.model_data['feature_cols']
        self.cat_cols = self.model_data['cat_cols']
        self.num_cols = self.model_data['num_cols']
        
    def predict(self, df):
        """Predict log_base_salary on new data."""
        # Use only the required feature columns
        X = df[self.feature_cols].copy()
        
        # Preprocess features
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions (log scale)
        log_predictions = self.model.predict(X_processed)
        
        # Transform back to original salary scale
        salary_predictions = np.exp(log_predictions)
        
        return salary_predictions, log_predictions
    
    def predict_with_confidence(self, df):
        """Predict with log-scale predictions for reference."""
        salaries, log_salaries = self.predict(df)
        
        result = df[['fiscal_year', 'base_salary']].copy()
        result['predicted_salary'] = salaries
        result['predicted_log_salary'] = log_salaries
        result['actual_log_salary'] = np.log(df['base_salary'])
        result['log_error'] = result['predicted_log_salary'] - result['actual_log_salary']
        result['salary_error_pct'] = (result['predicted_salary'] - df['base_salary']) / df['base_salary'] * 100
        
        return result


def main():
    """Example: Load model and predict on test data."""
    import sys
    sys.path.insert(0, '/Users/mehreen.gillaniicloud.com/Desktop/cuny,2025/second semester/622 ML/Final Project/NYC_Payroll_ML/data622')
    
    # Added your Codespace path
    sys.path.insert(0, '/workspaces/data622/src')

    from data622.dataset import load_salary_data, filter_model_population, add_tenure_proxy, split_by_year
    from data622.features import add_salary_target_features, add_feature_columns
    
    # --- Model Justification Summary ----------
    print("\n" + "="*45)
    print("MODEL SELECTION SUMMARY (R² SCORES)")
    print("="*45)
    print("➔ Linear Regression  : 0.7453 (SELECTED)")
    print("  Random Forest      : 0.7382")
    print("  Tuned XGBoost      : 0.7379")
    print("="*45 + "\n")
    # ------------------------------------------

    # Load linear regression model
    model_filename = 'salary_model_linear.pkl'
    print(f"Loading model: {model_filename}...")
    predictor = SalaryPredictor(MODELS_DIR / model_filename)
    
    # Load test data
    df = load_salary_data()
    df = filter_model_population(df)
    df = add_tenure_proxy(df)
    df = add_salary_target_features(df)
    df = add_feature_columns(df)
    
    _, _, test_df = split_by_year(df)
    
    # Make predictions
    print(f"\n📊 Testing Predictions on {len(test_df)} samples\n")
    results = predictor.predict_with_confidence(test_df)
    
    print("Sample Predictions (first 10 rows):")
    print(results[['fiscal_year', 'base_salary', 'predicted_salary', 'salary_error_pct']].head(10))
    
    print(f"\n📈 Prediction Accuracy Metrics:")
    print(f"  Mean Absolute Error:     {results['salary_error_pct'].abs().mean():.2f}%") #removed $ sign as its calculating percentage
    print(f"  Median Error:            {results['salary_error_pct'].median():.2f}%")
    print(f"  Std Dev:                 {results['salary_error_pct'].std():.2f}%")
    print(f"  Min Error:               {results['salary_error_pct'].min():.2f}%")
    print(f"  Max Error:               {results['salary_error_pct'].max():.2f}%")


if __name__ == '__main__':
    main()
