#!/usr/bin/env python3
# analyze_features.py - Run from NYC_Payroll_ML folder

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def get_feature_names(preprocessor, num_cols):
    """Extract feature names after one-hot encoding."""
    feature_names = []
    
    # Get numeric feature names (keep as is)
    feature_names.extend(num_cols)
    
    # Get categorical feature names (one-hot encoded)
    try:
        cat_transformer = preprocessor.named_transformers_['cat']
        onehot = cat_transformer.named_steps['onehot']
        cat_features = onehot.get_feature_names_out()
        feature_names.extend(cat_features)
    except Exception as e:
        print(f"Warning: Could not extract categorical features: {e}")
        # Fallback: create generic names
        n_cat_features = preprocessor.transformers_[0][2]  # Get categorical columns
        feature_names.extend([f"cat_{i}" for i in range(len(n_cat_features))])
    
    return feature_names

def analyze_feature_importance():
    """Analyze feature importance from trained models."""
    
    # Path to models
    models_dir = Path("data622/models")
    
    # Load the best model
    model_path = models_dir / "salary_model_linear_regression.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📂 Loading model: {model_path.name}")
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    num_cols = model_data.get('num_cols', [])
    cat_cols = model_data.get('cat_cols', [])
    
    # Get all feature names after preprocessing
    all_features = get_feature_names(preprocessor, num_cols)
    coefficients = model.coef_
    
    print(f"\n📊 Debug Info:")
    print(f"  Original numeric features: {len(num_cols)}")
    print(f"  Original categorical features: {len(cat_cols)}")
    print(f"  Expanded features (after one-hot): {len(all_features)}")
    print(f"  Coefficients: {len(coefficients)}")
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': all_features,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\n" + "="*80)
    print("📊 LINEAR REGRESSION FEATURE IMPORTANCE")
    print(f"Total Features: {len(importance_df)}")
    print("="*80)
    
    # Top 20 features
    print("\n🏆 TOP 20 MOST INFLUENTIAL FEATURES:")
    print("-"*80)
    for idx, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
        direction = "⬆️ INCREASES" if row['coefficient'] > 0 else "⬇️ DECREASES"
        # Clean up feature names for display
        feature_name = row['feature'].replace('title_std_grouped_', '').replace('agency_std_', '').replace('tenure_bucket_', '')
        print(f"{idx:2d}. {feature_name[:50]:50s} {direction:12s} (coef: {row['coefficient']:8.4f})")
    
    # Top positive features
    print("\n" + "="*80)
    print("📈 TOP 15 FEATURES THAT INCREASE SALARY:")
    print("-"*80)
    positive = importance_df[importance_df['coefficient'] > 0].head(15)
    for idx, (_, row) in enumerate(positive.iterrows(), 1):
        feature_name = row['feature'].replace('title_std_grouped_', '').replace('agency_std_', '')
        print(f"{idx:2d}. {feature_name[:55]:55s} +{row['coefficient']:8.4f}")
    
    # Top negative features
    print("\n" + "="*80)
    print("📉 TOP 15 FEATURES THAT DECREASE SALARY:")
    print("-"*80)
    negative = importance_df[importance_df['coefficient'] < 0].head(15)
    for idx, (_, row) in enumerate(negative.iterrows(), 1):
        feature_name = row['feature'].replace('title_std_grouped_', '').replace('agency_std_', '')
        print(f"{idx:2d}. {feature_name[:55]:55s} {row['coefficient']:8.4f}")
    
    # Summary by category
    print("\n" + "="*80)
    print("📂 FEATURE CATEGORY SUMMARY:")
    print("="*80)
    
    categories = {
        'Tenure': ['tenure'],
        'Year/Fiscal': ['fiscal_year', 'year'],
        'Title': ['title_'],
        'Agency': ['agency_'],
        'Salary Stats': ['median', 'avg', 'salary'],
        'Pay Basis': ['pay_basis', 'hourly'],
        'Location': ['borough'],
        'Employment': ['employment'],
        'OT/Hours': ['ot_', 'hours']
    }
    
    for category, keywords in categories.items():
        mask = importance_df['feature'].str.contains('|'.join(keywords), case=False, na=False)
        cat_features = importance_df[mask]
        if len(cat_features) > 0:
            print(f"\n{category}:")
            print(f"  • Features: {len(cat_features)}")
            print(f"  • Avg absolute coefficient: {cat_features['abs_coefficient'].mean():.4f}")
            print(f"  • Total impact: {cat_features['coefficient'].sum():.4f}")
            # Top feature in this category
            top = cat_features.iloc[0]
            direction = "+" if top['coefficient'] > 0 else ""
            print(f"  • Top: {top['feature'][:50]} ({direction}{top['coefficient']:.4f})")
    
    # Save full results
    output_path = models_dir / "feature_importance_full.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\n✅ Full analysis saved to: {output_path}")
    
    # Also save top 50 for quick reference
    top_50 = importance_df.head(50)
    top_50.to_csv(models_dir / "feature_importance_top50.csv", index=False)
    print(f"✅ Top 50 features saved to: {models_dir / 'feature_importance_top50.csv'}")

if __name__ == "__main__":
    analyze_feature_importance()