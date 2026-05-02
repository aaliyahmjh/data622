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

    # Target features
    df = add_salary_target_features(df)
    
    # Categorical grouping
    df = bucket_tenure(df)
    df = group_rare_titles(df)
    
    # Frequency & size features
    df = add_title_frequency(df)
    df = add_agency_size(df)

    return df


# Select only the columns used for modeling
def get_model_columns(df: pd.DataFrame):
    categorical_features = [
        c for c in [
            "agency_std",
            "title_std",
            "title_category",
            "title_std_grouped",
            "tenure_bucket",
            "pay_basis",
        ] if c in df.columns
    ]

    numeric_features = [
        c for c in [
            "fiscal_year",
            "tenure_years",
            "title_frequency",
            "agency_size",
            "median_salary_by_title",
            "median_salary_by_agency",
            "count_of_job_titles",
            "current_year",
            "regular_hours",
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

# static reference table from training/current-year data only
def build_reference_table(
    df: pd.DataFrame,
    current_year: int | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if current_year is None:
        current_year = int(df["fiscal_year"].max())

    current_df = df[df["fiscal_year"] == current_year].copy()

    title_stats = (
        current_df.groupby("title_std")
        .agg(
            median_salary_by_title=("base_salary", "median"),
            count_of_job_titles=("title_std", "size"),
        )
        .reset_index()
    )

    agency_stats = (
        current_df.groupby("agency_std")
        .agg(
            median_salary_by_agency=("base_salary", "median")
        )
        .reset_index()
    )

    title_agency_stats = (
        current_df.groupby(["agency_std", "title_std"])
        .agg(
            agency_title_count=("title_std", "size"),
            regular_hours=("regular_hours", "median"),
        )
        .reset_index()
    )

    ref = title_agency_stats.merge(title_stats, on="title_std", how="left")
    ref = ref.merge(agency_stats, on="agency_std", how="left")
    ref["current_year"] = current_year

    return ref

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


# Feature: overtime hours worked (proxy for workload/seniority) - Added
def add_ot_features(df):
    df = df.copy()

    if "ot_hours" in df.columns:
        df["ot_hours_total"] = df["ot_hours"].fillna(0)
        df["has_ot"] = (df["ot_hours_total"] > 0).astype(int)

    return df

# Feature:  merge reference features into model data
def add_reference_features(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    lookup_cols = [
        "agency_std",
        "title_std",
        "median_salary_by_title",
        "median_salary_by_agency",
        "count_of_job_titles",
        "agency_title_count",
        "regular_hours",
        "current_year",
    ]

    existing_lookup_cols = [c for c in lookup_cols if c in reference_df.columns]

    df = df.merge(
        reference_df[existing_lookup_cols],
        on=["agency_std", "title_std"],
        how="left",
    )

    return df

