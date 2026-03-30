##########################################
# config.py - Contains useful constants
##########################################

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCES_DIR = PROJECT_ROOT / "references"

DEFAULT_INPUT_FILE = PROCESSED_DATA_DIR / "nyc_annual_salary_employees_payBasis_perAnuum.csv"

TARGET_COL = "base_salary"
TARGET_LOG_COL = "log_base_salary"

CATEGORICAL_FEATURES = [
    "agency_std",
    "title_category",
    "title_std_grouped",
]

NUMERIC_FEATURES = [
    "fiscal_year",
    "tenure_years",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

BASE_COLUMNS = [
    "fiscal_year",
    "agency_std",
    "title_std",
    "title_category",
    "base_salary",
    "total_compensation",
    "regular_hours",
    "work_location_borough",
]

DROP_FOR_MODEL = [
    "total_compensation",
    "regular_hours",
]

MIN_BASE_SALARY = 10000
MAX_BASE_SALARY = 500000
MIN_TITLE_COUNT = 100

TRAIN_YEARS = list(range(2015, 2023))
VALID_YEARS = [2023]
TEST_YEARS = [2024]