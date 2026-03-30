##########################################
# dataset.py - Handles data loading + cleaning
##########################################
from __future__ import annotations

import pandas as pd
from pathlib import Path

from data622.paths import PROCESSED_DATA_DIR

# Default input file 
DEFAULT_INPUT_FILE = PROCESSED_DATA_DIR / "nyc_payroll_combined_all_2015_2024.csv"

# Salary bounds to remove extreme outliers
MIN_BASE_SALARY = 10000
MAX_BASE_SALARY = 500000

TRAIN_YEARS = list(range(2015, 2023))
VALID_YEARS = [2023]
TEST_YEARS = [2024]


# Convert column names to snake case
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


# Clean text fields (lowercase, trim, handle missing)
def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


# Load dataset and standardize column names
def load_salary_data(filepath: str | Path | None = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else DEFAULT_INPUT_FILE
    df = pd.read_csv(path, low_memory=False)
    df = standardize_column_names(df)
    return df


# Filter dataset to a clean, model-ready population
def filter_model_population(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Keep only relevant columns if they exist
    required_cols = [
        "fiscal_year",
        "agency_std",
        "title_std",
        "title_category",
        "base_salary",
        "total_compensation",
        "regular_hours",
        "work_location_borough",
        "ot_pay_ratio"
    ]
    existing_cols = [c for c in required_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Ensure valid salary values
    df["base_salary"] = pd.to_numeric(df["base_salary"], errors="coerce")
    df = df[df["base_salary"].notna()].copy()
    df = df[
        (df["base_salary"] >= MIN_BASE_SALARY)
        & (df["base_salary"] <= MAX_BASE_SALARY)
    ].copy()

    # Convert numeric columns
    if "fiscal_year" in df.columns:
        df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

    if "regular_hours" in df.columns:
        df["regular_hours"] = pd.to_numeric(df["regular_hours"], errors="coerce")

    # Clean categorical text fields
    for col in ["agency_std", "title_std", "title_category", "work_location_borough"]:
        if col in df.columns:
            df[col] = clean_text(df[col])

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# Create a proxy for tenure based on the first observed year per role
def add_tenure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = {"agency_std", "title_std", "fiscal_year"}
    if not needed.issubset(df.columns):
        df["tenure_years"] = 0
        return df

    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")

    # Find the first year each agency/title appears
    first_seen = (
        df.groupby(["agency_std", "title_std"], dropna=False)["fiscal_year"]
        .min()
        .reset_index(name="first_seen_year")
    )

    # Merge back and compute tenure
    df = df.merge(first_seen, on=["agency_std", "title_std"], how="left")

    if "first_seen_year" not in df.columns:
        df["tenure_years"] = 0
        return df

    df["tenure_years"] = (df["fiscal_year"] - df["first_seen_year"]).clip(lower=0)
    df = df.drop(columns=["first_seen_year"], errors="ignore")

    return df


# Split data into train/validation/test sets by year
def split_by_year(df: pd.DataFrame):
    train_df = df[df["fiscal_year"].isin(TRAIN_YEARS)].copy()
    valid_df = df[df["fiscal_year"].isin(VALID_YEARS)].copy()
    test_df = df[df["fiscal_year"].isin(TEST_YEARS)].copy()
    return train_df, valid_df, test_df


# Save train/validation/test to separate CSV files
def save_train_valid_test(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str = None
) -> tuple[Path, Path, Path]:
    """
    Save train, validation, and test sets to separate CSV files.
    
    Args:
        train_df: Training data
        valid_df: Validation data
        test_df: Test data
        output_dir: Directory to save CSVs (default: PROCESSED_DATA_DIR)
    
    Returns:
        Tuple of (train_path, valid_path, test_path)
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train_set.csv"
    valid_path = output_dir / "valid_set.csv"
    test_path = output_dir / "test_set.csv"
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train set saved: {train_path} ({len(train_df)} rows)")
    print(f"✅ Valid set saved: {valid_path} ({len(valid_df)} rows)")
    print(f"✅ Test set saved:  {test_path} ({len(test_df)} rows)")
    
    return train_path, valid_path, test_path
    return train_df, valid_df, test_df