##########################################
# features.py - Handles feature engineering
# SIMPLIFIED FOR ACTUAL AVAILABLE DATA
##########################################

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================
# CONSTANTS
# ============================================
MAX_TENURE_YEARS = 50  # Add this constant


# ============================================
# TARGET FEATURES
# ============================================

def add_salary_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-transformed salary target."""
    df = df.copy()
    df["log_base_salary"] = np.log(df["base_salary"].clip(lower=1))
    return df


# ============================================
# TENURE FEATURES
# ============================================

def bucket_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tenure into buckets."""
    df = df.copy()
    if "tenure_years" not in df.columns:
        df["tenure_years"] = 0
    
    df["tenure_bucket"] = pd.cut(
        df["tenure_years"],
        bins=[-np.inf, 0, 1, 3, 5, 10, 20, np.inf],
        labels=["0", "1", "2-3", "4-5", "6-10", "11-20", "20+"],
    ).astype(str)
    return df


# ============================================
# TITLE GROUPING
# ============================================

def group_rare_titles(df, source_col="title_std", output_col="title_std_grouped", 
                      min_count=100, title_counts_train=None):
    """Group rare job titles into 'other'."""
    df = df.copy()
    
    if title_counts_train is not None:
        common = title_counts_train[title_counts_train >= min_count].index
    else:
        counts = df[source_col].value_counts(dropna=False)
        common = counts[counts >= min_count].index
    
    df[output_col] = df[source_col].where(df[source_col].isin(common), "other_title")
    return df


# ============================================
# FREQUENCY & STATISTICS FEATURES
# ============================================

def add_title_frequency(df, title_counts_train=None):
    """Add frequency of each job title."""
    df = df.copy()
    if title_counts_train is not None:
        df["title_frequency"] = df["title_std"].map(title_counts_train).fillna(0)
    else:
        title_counts = df["title_std"].value_counts()
        df["title_frequency"] = df["title_std"].map(title_counts)
    return df


def add_agency_size(df, agency_counts_train=None):
    """Add number of employees per agency."""
    df = df.copy()
    if agency_counts_train is not None:
        df["agency_size"] = df["agency_std"].map(agency_counts_train).fillna(0)
    else:
        agency_counts = df["agency_std"].value_counts()
        df["agency_size"] = df["agency_std"].map(agency_counts)
    return df


def add_title_avg_salary(df, title_avg_train=None):
    """Add average salary per job title."""
    df = df.copy()
    if title_avg_train is not None:
        df["title_avg_salary"] = df["title_std"].map(title_avg_train)
        overall_avg = title_avg_train.mean() if not title_avg_train.empty else 0
        df["title_avg_salary"] = df["title_avg_salary"].fillna(overall_avg)
    else:
        title_avg = df.groupby("title_std")["base_salary"].mean()
        df["title_avg_salary"] = df["title_std"].map(title_avg)
    return df


# ============================================
# HOURS FEATURES (from dataset.py)
# ============================================

def add_hours_features(df):
    """Add features from regular_hours column."""
    df = df.copy()
    if "regular_hours" in df.columns:
        df["regular_hours_clean"] = pd.to_numeric(df["regular_hours"], errors="coerce").fillna(2080)
        df["is_full_time"] = (df["regular_hours_clean"] >= 35).astype(int)
        df["hours_missing"] = df["regular_hours"].isna().astype(int)
    else:
        df["regular_hours_clean"] = 2080
        df["is_full_time"] = 1
        df["hours_missing"] = 0
    return df


# ============================================
# OVERTIME FEATURES (from dataset.py columns)
# ============================================

def add_ot_features(df):
    """Add overtime features from existing columns."""
    df = df.copy()
    
    # OT hours
    if "ot_hours" in df.columns:
        df["ot_hours"] = df["ot_hours"].fillna(0)
        df["has_ot"] = (df["ot_hours"] > 0).astype(int)
    else:
        df["ot_hours"] = 0
        df["has_ot"] = 0
    
    # OT pay ratio (already in dataset.py as ot_pay_ratio)
    if "ot_pay_ratio" in df.columns:
        df["ot_pay_ratio"] = df["ot_pay_ratio"].fillna(0)
    else:
        df["ot_pay_ratio"] = 0
    
    # Total OT paid
    if "total_ot_paid" in df.columns:
        df["total_ot_paid"] = df["total_ot_paid"].fillna(0)
    
    return df


# ============================================
# OTHER PAY FEATURES
# ============================================

def add_other_pay_features(df):
    """Add features from total_other_pay column."""
    df = df.copy()
    
    if "total_other_pay" in df.columns:
        df["total_other_pay"] = df["total_other_pay"].fillna(0)
        df["has_other_pay"] = (df["total_other_pay"] > 0).astype(int)
        if "base_salary" in df.columns:
            df["other_pay_ratio"] = (df["total_other_pay"] / df["base_salary"].clip(lower=1)).clip(upper=0.5)
        else:
            df["other_pay_ratio"] = 0
    else:
        df["has_other_pay"] = 0
        df["other_pay_ratio"] = 0
    
    return df


# ============================================
# COMPENSATION FEATURES
# ============================================

def add_compensation_features(df):
    """Add features from total_compensation."""
    df = df.copy()
    if "total_compensation" in df.columns and "base_salary" in df.columns:
        df["total_compensation"] = pd.to_numeric(df["total_compensation"], errors="coerce")
        df["comp_ratio"] = (df["total_compensation"] / df["base_salary"].clip(lower=1)).clip(upper=3)
    else:
        df["comp_ratio"] = 1.0
    return df


# ============================================
# MAIN FEATURE ENGINEERING FUNCTION
# ============================================

def add_feature_columns(df, title_counts_train=None, agency_counts_train=None, title_avg_train=None):
    """Add all engineered features in one step."""
    df = df.copy()
    
    # Target feature
    df = add_salary_target_features(df)
    
    # Tenure features
    df = bucket_tenure(df)
    
    # Title grouping
    df = group_rare_titles(df, title_counts_train=title_counts_train)
    
    # Frequency features
    df = add_title_frequency(df, title_counts_train=title_counts_train)
    df = add_agency_size(df, agency_counts_train=agency_counts_train)
    df = add_title_avg_salary(df, title_avg_train=title_avg_train)
    
    # Hours features
    df = add_hours_features(df)
    
    # OT features
    df = add_ot_features(df)
    
    # Other pay features
    df = add_other_pay_features(df)
    
    # Compensation features
    df = add_compensation_features(df)
    
    # Note: pay_basis features removed since data is already annual only
    
    return df


# ============================================
# MODEL COLUMNS SELECTION
# ============================================

def get_model_columns(df: pd.DataFrame):
    """Return only the features that actually exist in the dataframe."""
    
    categorical_features = [
        c for c in [
            "agency_std",
            "title_category",
            "title_std_grouped",
            "tenure_bucket",
            "is_full_time",
            "has_ot",
            "has_other_pay"
        ] if c in df.columns
    ]
    
    numeric_features = [
        c for c in [
            "fiscal_year",
            "tenure_years",
            "title_frequency",
            "agency_size",
            "title_avg_salary",
            "regular_hours_clean",
            "hours_missing",
            "ot_hours",
            "ot_pay_ratio",
            "total_ot_paid",
            "other_pay_ratio",
            "comp_ratio",
            "log_base_salary"
        ] if c in df.columns
    ]
    
    return categorical_features, numeric_features


# ============================================
# PREPROCESSING PIPELINE
# ============================================

def make_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline."""
    categorical_features, numeric_features = get_model_columns(df)
    
    if len(categorical_features) == 0 and len(numeric_features) == 0:
        raise ValueError("No features found! Check get_model_columns function.")
    
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Numeric features: {len(numeric_features)}")
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_features),
        ("num", numeric_pipeline, numeric_features)
    ], remainder="drop")
    
    return preprocessor


# ============================================
# TRAINING STATISTICS FUNCTIONS
# ============================================

def fit_feature_stats(train_df: pd.DataFrame):
    """Fit statistics on training data only."""
    stats = {
        'title_counts': train_df['title_std'].value_counts(),
        'agency_counts': train_df['agency_std'].value_counts(),
        'title_avg_salary': train_df.groupby('title_std')['base_salary'].mean()
    }
    return stats


def transform_with_stats(df: pd.DataFrame, stats: dict):
    """Apply feature engineering using training statistics."""
    df = df.copy()
    
    df = add_salary_target_features(df)
    df = bucket_tenure(df)
    df = group_rare_titles(df, title_counts_train=stats['title_counts'])
    df = add_title_frequency(df, title_counts_train=stats['title_counts'])
    df = add_agency_size(df, agency_counts_train=stats['agency_counts'])
    df = add_title_avg_salary(df, title_avg_train=stats['title_avg_salary'])
    df = add_hours_features(df)
    df = add_ot_features(df)
    df = add_other_pay_features(df)
    df = add_compensation_features(df)
    
    return df