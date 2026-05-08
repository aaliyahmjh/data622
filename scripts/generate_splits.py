#!/usr/bin/env python3
"""Generate train/valid/test CSV files from processed dataset."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

# Use relative imports from the package
from data622.src.data622.dataset import (
    load_salary_data,
    filter_model_population,
    add_tenure_proxy,
    add_ot_features,
    add_compensation_features,
    split_by_year
)


def main():
    """Main function to generate splits."""
    print("\n" + "="*60)
    print("GENERATING TRAIN/VALID/TEST SPLITS")
    print("="*60)
    
    # Load and process data
    print("\n📂 Loading data...")
    df = load_salary_data()
    
    print("\n🔍 Filtering model population...")
    df = filter_model_population(df)
    
    print("\n📊 Adding tenure proxy...")
    df = add_tenure_proxy(df)
    
    print("\n➕ Adding OT features...")
    df = add_ot_features(df)
    
    print("\n💰 Adding compensation features...")
    df = add_compensation_features(df)
    
    # Split by year
    print("\n✂️  Splitting by year...")
    train_df, valid_df, test_df = split_by_year(df)
    
    # Save to separate CSVs
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving files...")
    train_df.to_csv(output_dir / "train_set.csv", index=False)
    valid_df.to_csv(output_dir / "valid_set.csv", index=False)
    test_df.to_csv(output_dir / "test_set.csv", index=False)
    
    print("\n" + "="*60)
    print("✅ FILES CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"   Train set: {len(train_df):,} rows → data/processed/train_set.csv")
    print(f"   Valid set: {len(valid_df):,} rows → data/processed/valid_set.csv")
    print(f"   Test set:  {len(test_df):,} rows → data/processed/test_set.csv")
    print("="*60)


if __name__ == "__main__":
    main()