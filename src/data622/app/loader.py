from pathlib import Path

import pandas as pd
import streamlit as st

from data622.app.config import (
    DATA_DICT_FILE,
    MODEL_FILE,
    REFERENCE_TABLE_FILE,
    TEST_SET_FILE,
    TRAIN_SET_FILE,
    VALID_SET_FILE,
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
    """
    Load a lightweight year-over-year salary summary across all data splits.

    Combines train (2015-2022), valid (2023), and test (2024) sets so the
    historical chart has no gap before the projection.
    Returns median base_salary grouped by fiscal_year, title_std, agency_std.
    """
    cols = ["fiscal_year", "title_std", "agency_std", "base_salary", "regular_hours"]
    frames = []
    for path in (TRAIN_SET_FILE, VALID_SET_FILE, TEST_SET_FILE):
        if Path(path).exists():
            frames.append(pd.read_csv(path, usecols=cols, low_memory=False))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["fiscal_year", "title_std", "agency_std"])
        .agg(
            base_salary=("base_salary", "median"),
            headcount=("base_salary", "count"),
            regular_hours=("regular_hours", "median"),
        )
        .reset_index()
    )
    return summary


@st.cache_data(show_spinner="Loading title categories…")
def load_title_category_map() -> dict[str, str]:
    """
    Build a title_std -> title_category lookup from train_set.csv.

    Returns a dict mapping each standardized title to its most common category.
    Falls back to empty dict if the file is missing.
    """
    path = Path(TRAIN_SET_FILE)
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["title_std", "title_category"], low_memory=False)
    mapping = (
        df.dropna(subset=["title_std", "title_category"])
        .groupby("title_std")["title_category"]
        .agg(lambda x: x.value_counts().index[0])
    )
    return mapping.to_dict()


@st.cache_data(show_spinner="Loading data dictionary…")
def load_data_dictionary() -> pd.DataFrame | None:
    """Load the data dictionary CSV from the references directory."""
    dict_path = Path(DATA_DICT_FILE)
    if not dict_path.exists():
        return None
    return pd.read_csv(dict_path)
