##########################################
# train.py - Training on processed data
##########################################

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from data622.paths import MODELS_DIR, PROCESSED_DATA_DIR

from data622.dataset import (
    load_salary_data,
    filter_model_population,
    add_tenure_proxy,
    split_by_year,
)
from data622.features import (
    add_feature_columns,
    build_reference_table,
    add_reference_features,
    get_model_columns,
    make_preprocessor,
)

def main():
    print("Loading data...")
    df = load_salary_data()

    print("Cleaning data...")
    df = filter_model_population(df)
    df = add_tenure_proxy(df)

    print("Splitting data...")
    train_df, valid_df, test_df = split_by_year(df)

    print("Adding base features...")
    train_df = add_feature_columns(train_df)
    valid_df = add_feature_columns(valid_df)
    test_df = add_feature_columns(test_df)

    print("Building reference table from train only...")
    reference_df = build_reference_table(train_df)

    print("Adding reference features...")
    train_df = add_reference_features(train_df, reference_df)
    valid_df = add_reference_features(valid_df, reference_df)
    test_df = add_reference_features(test_df, reference_df)

    print("Checking model columns...")
    cat_cols, num_cols = get_model_columns(train_df)
    feature_cols = cat_cols + num_cols

    preprocessor = make_preprocessor(train_df)
    X_train = train_df[feature_cols]
    preprocessor.fit_transform(X_train)

    print("Saving train/valid/test files...")
    train_df.to_csv(PROCESSED_DATA_DIR / "train_set.csv", index=False)
    valid_df.to_csv(PROCESSED_DATA_DIR / "valid_set.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_set.csv", index=False)

    reference_df.to_csv(PROCESSED_DATA_DIR / "reference_table.csv", index=False)

    print("Pipeline complete")


if __name__ == "__main__":
    main()