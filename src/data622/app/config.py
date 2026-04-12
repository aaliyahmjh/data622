from data622.config import *  # noqa: F401, F403 - re-export project-level constants
from data622.paths import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REFERENCES_DIR,
)

# Processed Data files
DATA_FILE = PROCESSED_DATA_DIR / "dummy_data.csv"

# Model file
MODEL_FILE = MODELS_DIR / "salary_model.pkl"


DATA_DICT_FILE = REFERENCES_DIR / "data_dictionary.csv"


# Column name mapping
# update strings with expected names
COL_JOB_TITLE = "title_description"
COL_AGENCY = "agency_name"
COL_BOROUGH = "work_location_borough"
COL_SALARY = "base_salary"
COL_FISCAL_YEAR = "fiscal_year"
COL_YEARS_SERVICE = "years_of_service"
COL_PAY_BASIS = "pay_basis"


BOROUGHS = ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]

# Consistent borough colors used across all visualizations.
# Hex values for Altair; RGBA lists (0-255) for pydeck.
BOROUGH_COLORS_HEX: dict[str, str] = {
    "MANHATTAN":     "#0d6e7a",  # teal
    "BROOKLYN":      "#5b4a9e",  # purple
    "QUEENS":        "#b84040",  # red
    "BRONX":         "#e8557a",  # pink
    "STATEN ISLAND": "#e8a830",  # amber
}

BOROUGH_COLORS_RGBA: dict[str, list[int]] = {
    "MANHATTAN":     [ 13, 110, 122, 200],
    "BROOKLYN":      [ 91,  74, 158, 200],
    "QUEENS":        [184,  64,  64, 200],
    "BRONX":         [232,  85, 122, 200],
    "STATEN ISLAND": [232, 168,  48, 200],
}

# Set year ranges for charts
YEAR_MIN = 2015
YEAR_MAX = 2024

# Model features - what we expect users to input for predictions
MODEL_FEATURE_COLS = [
    COL_JOB_TITLE,
    COL_AGENCY,
    COL_BOROUGH,
    COL_YEARS_SERVICE,
]

# App constants
APP_TITLE = "NYC Payroll Dashboard"
APP_ICON = "https://www.nyc.gov/assets/dcas/images/content/pages/CitySealColor.jpg"
APP_SUBTITLE = "*Dictate your career with the City of New York*"
APP_DISCLAIMER = (
    "This website is not associated with the City of New York. "
    "All data is publicly available from www.nyc.gov"
)
