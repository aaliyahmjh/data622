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


# Add year-relative features (Option 3) to handle salary growth across years
def add_year_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create salary features relative to year average - handles temporal drift."""
    df = df.copy()
    
    # Calculate salary statistics by fiscal year
    year_stats = df.groupby('fiscal_year')['base_salary'].agg([
        ('year_mean_salary', 'mean'),
        ('year_median_salary', 'median'),
        ('year_std_salary', 'std')
    ]).reset_index()
    
    # Merge year statistics back to main dataframe
    df = df.merge(year_stats, on='fiscal_year', how='left')
    
    # Feature 1: Salary relative to year average (ratio)
    df['salary_vs_year_mean'] = df['base_salary'] / df['year_mean_salary'].clip(lower=1)
    
    # Feature 2: Salary relative to year median (ratio)
    df['salary_vs_year_median'] = df['base_salary'] / df['year_median_salary'].clip(lower=1)
    
    # Feature 3: Percentile ranking within year (0-1)
    df['salary_percentile_by_year'] = df.groupby('fiscal_year')['base_salary'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Feature 4: Z-score within year (how many std devs from year average)
    df['salary_z_score_by_year'] = (
        (df['base_salary'] - df['year_mean_salary']) / df['year_std_salary'].clip(lower=0.1)
    ).clip(-3, 3)  # Clip extreme outliers
    
    # Feature 5: Salary deviation from year mean (in dollars)
    df['salary_deviation_from_year_mean'] = df['base_salary'] - df['year_mean_salary']
    
    return df


# Add all engineered features in one step
def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target features
    df = add_salary_target_features(df)
    
    # Categorical grouping
    df = bucket_tenure(df)
    df = group_rare_titles(df)
    
    # Year-relative features (NEW - Option 3)
    df = add_year_relative_features(df)
    
    # Frequency & size features
    df = add_title_frequency(df)
    df = add_agency_size(df)
    df = add_title_avg_salary(df)
    
    # NEW: overtime features (only if ot_hours exists)
    if "ot_hours" in df.columns:
        df = add_ot_features(df)

    return df


# Select only the columns used for modeling
def get_model_columns(df: pd.DataFrame):
    categorical_features = [
        c for c in [
            "agency_std", 
            "title_category", 
            "title_std_grouped",
            "tenure_bucket",
            "borough_std",
            "employment_type_std"
        ] if c in df.columns
    ]
    numeric_features = [
        c for c in [
            "fiscal_year",
            "tenure_years",
            "title_frequency",
            "agency_size",
            "title_avg_salary",
            # NEW: Year-relative features (Option 3)
            "salary_vs_year_mean",
            "salary_vs_year_median",
            "salary_percentile_by_year",
            "salary_z_score_by_year",
            "salary_deviation_from_year_mean",
            # Existing OT & Hours features
            "ot_hours_total",
            "has_ot",
            "ot_pay_ratio",
            "regular_hours_worked",
            "is_full_time_hours",
            "hours_reliable",
            # NEW: Pay composition
            "other_pay_amount",
            "has_other_pay",
            "pay_other_ratio",
            "gross_to_base_ratio",
            # NEW: Agency stats
            "agency_median_salary",
            "agency_salary_std",
            "salary_percentile_in_agency",
            # NEW: Title stats
            "title_median_salary",
            "title_min_salary",
            "title_max_salary",
            "salary_deviation_from_title",
            # NEW: Temporal
            "years_since_start",
            "year_avg_salary",
            "is_full_year"
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



# Feature: overtime hours worked (proxy for workload/seniority) - Added
def add_ot_features(df):
    df = df.copy()
    
    # Total OT hours (many nulls, so fill with 0)
    df["ot_hours_total"] = df["ot_hours"].fillna(0)
    
    # Whether employee has OT (binary indicator)
    df["has_ot"] = (df["ot_hours_total"] > 0).astype(int)
    
    # OT pay ratio (OT paid / base salary) - workers with high OT get premium
    df["ot_pay_ratio"] = (df["total_ot_paid"].fillna(0) / df["base_salary"].clip(lower=1)).clip(upper=1.0)
    
    return df




