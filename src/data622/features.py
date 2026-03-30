##########################################
# features.py - Handles feature engineering
##########################################
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Add log-transformed salary (for skewed distribution)
def add_salary_target_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_base_salary"] = np.log(df["base_salary"].clip(lower=1))
    return df


# Convert tenure into buckets for easier modeling
def bucket_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tenure_years" not in df.columns:
        df["tenure_years"] = 0

    df["tenure_bucket"] = pd.cut(
        df["tenure_years"],
        bins=[-np.inf, 0, 1, 3, 5, 10, 20, np.inf],
        labels=["0", "1", "2-3", "4-5", "6-10", "11-20", "20+"],
    ).astype(str)

    return df


# Group infrequent job titles into "other" to reduce sparsity
def group_rare_titles(
    df: pd.DataFrame,
    source_col: str = "title_std",
    output_col: str = "title_std_grouped",
    min_count: int = 100,
) -> pd.DataFrame:
    df = df.copy()

    counts = df[source_col].value_counts(dropna=False)
    common = counts[counts >= min_count].index
    df[output_col] = df[source_col].where(df[source_col].isin(common), "other_title")

    return df


# Add all engineered features in one step
def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = add_salary_target_features(df)
    df = bucket_tenure(df)
    df = group_rare_titles(df)

    df = add_title_frequency(df)
    df = add_agency_size(df)
    df = add_title_avg_salary(df)

    return df


# Select only the columns used for modeling
def get_model_columns(df: pd.DataFrame):
    categorical_features = [
        c for c in ["agency_std", "title_category", "title_std_grouped"] if c in df.columns
    ]
    numeric_features = [
        c for c in [
            "fiscal_year",
            "tenure_years",
            "title_frequency",
            "agency_size",
            "title_avg_salary"
        ] if c in df.columns
    ]
    return categorical_features, numeric_features


# Build preprocessing pipeline
def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_features, numeric_features = get_model_columns(df)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    return preprocessor


# Feature: how common each job title is
def add_title_frequency(df):
    df = df.copy()
    title_counts = df["title_std"].value_counts()
    df["title_frequency"] = df["title_std"].map(title_counts)
    return df


# Feature: number of employees per agency
def add_agency_size(df):
    df = df.copy()
    agency_counts = df["agency_std"].value_counts()
    df["agency_size"] = df["agency_std"].map(agency_counts)
    return df


# Feature: average salary per job title 
def add_title_avg_salary(df):
    df = df.copy()
    title_avg = df.groupby("title_std")["base_salary"].mean()
    df["title_avg_salary"] = df["title_std"].map(title_avg)
    return df