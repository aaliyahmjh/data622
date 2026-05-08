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
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DEFAULT_INPUT_FILE = PROCESSED_DATA_DIR / "nyc_annual_salary_employees_payBasis_perAnuum.csv"

TARGET_COL = "base_salary"
TARGET_LOG_COL = "log_base_salary"

# ============================================
# FEATURE DEFINITIONS
# ============================================

CATEGORICAL_FEATURES = [
    "agency_std",
    "title_category",
    "title_std_grouped",  # Grouped rare titles
    "tenure_bucket",      # Bucketed tenure
    "borough_std",        # Standardized borough
    "employment_type_std", # Standardized employment type
    "pay_basis",          # Annual or hourly
    "is_hourly",          # Binary indicator
    "is_full_time"        # Binary indicator
]

NUMERIC_FEATURES = [
    # User input features
    "fiscal_year",
    "tenure_years",
    "regular_hours_worked",
    
    # Lookup-based features (from training)
    "title_median_salary",
    "agency_median_salary",
    "title_frequency",
    "year_avg_salary",
    
    # Derived features
    "log_base_salary",
    "years_since_start",
    "calculated_salary",
    "title_avg_salary",
    "agency_size",
    
    # OT features
    "ot_hours_total",
    "has_ot",
    "ot_pay_ratio",
    
    # Pay composition
    "other_pay_amount",
    "has_other_pay",
    "pay_other_ratio",
    "gross_to_base_ratio",
    
    # Agency stats
    "agency_salary_std",
    "salary_percentile_in_agency",
    
    # Title stats
    "title_min_salary",
    "title_max_salary",
    "salary_deviation_from_title",
    
    # Temporal
    "is_full_year"
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ============================================
# COLUMN MAPPINGS
# ============================================

BASE_COLUMNS = [
    "fiscal_year",
    "agency_std",
    "title_std",
    "title_category",
    "base_salary",
    "total_compensation",
    "regular_hours",
    "work_location_borough",
    "tenure_years",           # Added
    "pay_basis",              # Added
    "ot_hours",               # Added
    "total_ot_paid"           # Added
]

DROP_FOR_MODEL = [
    "total_compensation",
    "regular_hours",
    "base_salary",            # Keep separate as target
    "ot_hours",               # Already engineered
    "total_ot_paid"           # Already engineered
]

# ============================================
# DATA FILTERING CONSTANTS
# ============================================

MIN_BASE_SALARY = 10000
MAX_BASE_SALARY = 500000
MIN_TITLE_COUNT = 100
MIN_AGENCY_SIZE = 10          # Minimum employees per agency
MAX_TENURE_YEARS = 50         # Cap for extreme tenure values

# ============================================
# TRAIN/VALID/TEST SPLIT YEARS
# ============================================

TRAIN_YEARS = list(range(2015, 2023))   # 2015-2022 for training
VALID_YEARS = [2023]                     # 2023 for validation
TEST_YEARS = [2024]                      # 2024 for testing

# ============================================
# FEATURE ENGINEERING CONSTANTS
# ============================================

# Tenure buckets (years)
TENURE_BINS = [-np.inf, 0, 1, 3, 5, 10, 20, np.inf]  # Note: needs np imported
TENURE_LABELS = ["0", "1", "2-3", "4-5", "6-10", "11-20", "20+"]

# Regular hours thresholds (for full-time classification)
FULL_TIME_HOURS_THRESHOLD = 35

# OT ratio cap (to prevent extreme outliers)
MAX_OT_RATIO = 1.0

# Rare title grouping
RARE_TITLE_MIN_COUNT = 100

# ============================================
# MODEL HYPERPARAMETERS
# ============================================

# Random Forest defaults
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None
RF_RANDOM_STATE = 42

# XGBoost defaults  
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1
XGB_MAX_DEPTH = 5
XGB_RANDOM_STATE = 42

# Ridge/Lasso
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.001
ELASTICNET_ALPHA = 0.001

# Cross-validation
DEFAULT_CV_FOLDS = 5
RANDOM_SEARCH_ITER = 5

# ============================================
# PATHS FOR SAVING ARTIFACTS
# ============================================

def get_model_path(model_name: str) -> Path:
    """Get path for saving model artifacts."""
    return MODELS_DIR / f"salary_model_{model_name}.pkl"

def get_feature_stats_path() -> Path:
    """Get path for saving feature statistics."""
    return MODELS_DIR / "feature_stats.pkl"

def get_preprocessor_path() -> Path:
    """Get path for saving preprocessor."""
    return MODELS_DIR / "preprocessor.pkl"