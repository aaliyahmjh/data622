#!/usr/bin/env python3
"""Generate train/valid/test CSV files from processed dataset."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from pathlib import Path
from data622.dataset import (
    load_salary_data,
    filter_model_population,
    add_tenure_proxy,
    split_by_year
)
from data622.features import add_feature_columns

# Load and process data
print("Loading data...")
df = load_salary_data()
df = filter_model_population(df)
df = add_tenure_proxy(df)
df = add_feature_columns(df)

# Split by year
print("Splitting data...")
train_df, valid_df, test_df = split_by_year(df)

# Save to separate CSVs
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

print("Saving files...")
train_df.to_csv(output_dir / "train_set.csv", index=False)
valid_df.to_csv(output_dir / "valid_set.csv", index=False)
test_df.to_csv(output_dir / "test_set.csv", index=False)

print("\n✅ Files created successfully!")
print(f"   Train set: {len(train_df)} rows → data/processed/train_set.csv")
print(f"   Valid set: {len(valid_df)} rows → data/processed/valid_set.csv")
print(f"   Test set:  {len(test_df)} rows → data/processed/test_set.csv")
