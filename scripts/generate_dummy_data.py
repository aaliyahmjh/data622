"""
generate_dummy_data.py

Creates realistic dummy data for the MVP demo:
  - data/processed/data.csv          (payroll records)
  - references/data_dictionary.csv   (column descriptions)

Run with:
    uv run python scripts/generate_dummy_data.py
"""

import numpy as np
import pandas as pd

from data622.paths import PROCESSED_DATA_DIR, REFERENCES_DIR

SEED = 42
N_ROWS = 5_000

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Reference tables — representative NYC job titles and agencies
# ---------------------------------------------------------------------------
JOB_TITLES = [
    "POLICE OFFICER",
    "TEACHER",
    "FIREFIGHTER",
    "CIVIL ENGINEER",
    "SOCIAL WORKER",
    "STAFF ANALYST",
    "ADMINISTRATIVE ASSISTANT",
    "SANITATION WORKER",
    "NURSE",
    "ACCOUNTANT",
    "COMPUTER SYSTEMS MANAGER",
    "CORRECTION OFFICER",
    "PARKS WORKER",
    "ATTORNEY",
    "BUILDING INSPECTOR",
]

AGENCIES = [
    "POLICE DEPARTMENT",
    "DEPT OF EDUCATION",
    "FIRE DEPARTMENT",
    "DEPT OF TRANSPORTATION",
    "DEPT OF SOCIAL SERVICES",
    "OFFICE OF MANAGEMENT & BUDGET",
    "DEPT OF SANITATION",
    "HEALTH AND HOSPITALS CORP",
    "DEPT OF FINANCE",
    "DEPT OF PARKS & RECREATION",
]

BOROUGHS = ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]

PAY_BASIS = ["per Annum", "per Annum", "per Annum", "per Hour", "per Day"]

# Base salary ranges (mean, std) per job title — rough NYC approximations
SALARY_PARAMS: dict[str, tuple[float, float]] = {
    "POLICE OFFICER":            (75_000,  12_000),
    "TEACHER":                   (72_000,  10_000),
    "FIREFIGHTER":               (80_000,  11_000),
    "CIVIL ENGINEER":            (90_000,  15_000),
    "SOCIAL WORKER":             (60_000,   8_000),
    "STAFF ANALYST":             (68_000,  10_000),
    "ADMINISTRATIVE ASSISTANT":  (52_000,   7_000),
    "SANITATION WORKER":         (65_000,   9_000),
    "NURSE":                     (88_000,  13_000),
    "ACCOUNTANT":                (75_000,  11_000),
    "COMPUTER SYSTEMS MANAGER":  (105_000, 18_000),
    "CORRECTION OFFICER":        (70_000,  10_000),
    "PARKS WORKER":              (48_000,   6_000),
    "ATTORNEY":                  (98_000,  20_000),
    "BUILDING INSPECTOR":        (72_000,  10_000),
}

# Borough salary multiplier (Manhattan pays a bit more, Bronx a bit less)
BOROUGH_MULTIPLIER: dict[str, float] = {
    "MANHATTAN":     1.08,
    "BROOKLYN":      1.00,
    "QUEENS":        0.98,
    "BRONX":         0.95,
    "STATEN ISLAND": 0.97,
}

# ---------------------------------------------------------------------------
# Generate records
# ---------------------------------------------------------------------------
fiscal_years = rng.integers(2015, 2025, size=N_ROWS)  # 2015–2024 inclusive
titles = rng.choice(JOB_TITLES, size=N_ROWS)
agencies = rng.choice(AGENCIES, size=N_ROWS)
boroughs = rng.choice(BOROUGHS, size=N_ROWS)
years_of_service = rng.integers(0, 35, size=N_ROWS)
pay_basis = rng.choice(PAY_BASIS, p=[0.7, 0.1, 0.1, 0.05, 0.05], size=N_ROWS)

# Salary: base from title params + experience premium + borough multiplier
base_salaries = np.array([
    rng.normal(SALARY_PARAMS[t][0], SALARY_PARAMS[t][1])
    for t in titles
])
experience_premium = years_of_service * rng.uniform(500, 1_500, size=N_ROWS)
borough_mult = np.array([BOROUGH_MULTIPLIER[b] for b in boroughs])
salaries = np.clip((base_salaries + experience_premium) * borough_mult, 30_000, 250_000)

df = pd.DataFrame({
    "title_description":       titles,
    "agency_name":             agencies,
    "work_location_borough":   boroughs,
    "base_salary":             salaries.round(2),
    "fiscal_year":             fiscal_years,
    "years_of_service":        years_of_service,
    "pay_basis":               pay_basis,
})

out_path = PROCESSED_DATA_DIR / "data.csv"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"Wrote {len(df):,} rows to {out_path}")

# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------
data_dict = pd.DataFrame([
    {
        "column_name": "title_description",
        "data_type":   "string",
        "description": "Standardized job title for the employee.",
        "example":     "POLICE OFFICER",
    },
    {
        "column_name": "agency_name",
        "data_type":   "string",
        "description": "NYC agency that employs the worker.",
        "example":     "POLICE DEPARTMENT",
    },
    {
        "column_name": "work_location_borough",
        "data_type":   "string",
        "description": "NYC borough where the employee is based.",
        "example":     "MANHATTAN",
    },
    {
        "column_name": "base_salary",
        "data_type":   "float",
        "description": "Annual base salary in USD before overtime or other pay.",
        "example":     "75000.00",
    },
    {
        "column_name": "fiscal_year",
        "data_type":   "integer",
        "description": "NYC fiscal year the record corresponds to (2015–2024).",
        "example":     "2022",
    },
    {
        "column_name": "years_of_service",
        "data_type":   "integer",
        "description": "Number of full years the employee has been with the city.",
        "example":     "7",
    },
    {
        "column_name": "pay_basis",
        "data_type":   "string",
        "description": "How the employee is compensated (per Annum, per Hour, per Day).",
        "example":     "per Annum",
    },
])

dict_path = REFERENCES_DIR / "data_dictionary.csv"
REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
data_dict.to_csv(dict_path, index=False)
print(f"Wrote data dictionary ({len(data_dict)} rows) to {dict_path}")
