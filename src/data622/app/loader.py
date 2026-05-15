import json
from pathlib import Path

import pandas as pd
import streamlit as st

from data622.app.config import (
    DATA_DICT_FILE,
    MODEL_FILE,
    REFERENCE_TABLE_FILE,
    TITLE_CATEGORY_MAP_FILE,
    YOY_SUMMARY_FILE,
)
from data622.predict import SalaryPredictor


@st.cache_resource(show_spinner="Loading model…")
def load_model() -> SalaryPredictor | None:
    """Load the trained SalaryPredictor from disk, or None if not found."""
    path = Path(MODEL_FILE)
    if not path.exists():
        return None
    return SalaryPredictor(path)


@st.cache_data(show_spinner="Loading reference table…")
def load_reference_table() -> pd.DataFrame | None:
    """Load the static reference table of aggregated salary stats by title/agency."""
    path = Path(REFERENCE_TABLE_FILE)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading salary history…")
def load_yoy_summary() -> pd.DataFrame | None:
    """Load the pre-computed year-over-year salary summary."""
    path = Path(YOY_SUMMARY_FILE)
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading title categories…")
def load_title_category_map() -> dict[str, str]:
    """Load the pre-computed title_std -> title_category mapping."""
    path = Path(TITLE_CATEGORY_MAP_FILE)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading data dictionary…")
def load_data_dictionary() -> pd.DataFrame | None:
    """Load the data dictionary CSV from the references directory."""
    dict_path = Path(DATA_DICT_FILE)
    if not dict_path.exists():
        return None
    return pd.read_csv(dict_path)
