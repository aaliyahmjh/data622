import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from data622.app.config import (
    COL_BOROUGH,
    COL_FISCAL_YEAR,
    COL_JOB_TITLE,
    COL_SALARY,
    COL_YEARS_SERVICE,
    DATA_DICT_FILE,
    DATA_FILE,
    MODEL_FILE,
    YEAR_MAX,
    YEAR_MIN,
)


@st.cache_data(show_spinner="Loading payroll data…")
def load_payroll_data() -> pd.DataFrame:
    """
    Load and minimally prepare the processed payroll dataset.

    Returns a DataFrame filtered to the configured fiscal year range with
    only the columns needed by the app. Rows with null salary or borough
    are dropped.

    Partners: replace DATA_FILE in config.py to point to your dataset.
    Your dataset must contain the columns defined by COL_* in config.py.
    """
    needed_cols = [
        COL_JOB_TITLE,
        COL_SALARY,
        COL_BOROUGH,
        COL_FISCAL_YEAR,
        COL_YEARS_SERVICE,
    ]

    df = pd.read_csv(DATA_FILE, usecols=needed_cols, low_memory=False)

    # Normalise types
    df[COL_FISCAL_YEAR] = pd.to_numeric(df[COL_FISCAL_YEAR], errors="coerce")
    df[COL_SALARY] = pd.to_numeric(df[COL_SALARY], errors="coerce")
    df[COL_BOROUGH] = df[COL_BOROUGH].str.strip().str.upper()

    # Filter to configured year range and drop unusable rows
    df = df[df[COL_FISCAL_YEAR].between(YEAR_MIN, YEAR_MAX)]
    df = df.dropna(subset=[COL_SALARY, COL_BOROUGH])
    df = df[df[COL_SALARY] > 0]

    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the trained salary model from disk.

    Returns the model object, or None if no model file exists yet.

    Partners: save your trained model as a .pkl file to the path defined
    by MODEL_FILE in config.py.  The model must expose a .predict() method
    that accepts a DataFrame with columns matching MODEL_FEATURE_COLS.
    """
    model_path = Path(MODEL_FILE)
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading data dictionary…")
def load_data_dictionary() -> pd.DataFrame | None:
    """
    Load the data dictionary CSV from the references directory.

    Expected columns: column_name, data_type, description, example
    Returns None if the file does not exist.

    Partners: edit references/data_dictionary.csv to document your columns.
    """
    dict_path = Path(DATA_DICT_FILE)
    if not dict_path.exists():
        return None
    return pd.read_csv(dict_path)
