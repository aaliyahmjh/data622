##########################################
# paths.py - Defines commonly used paths
##########################################

from pathlib import Path

# Project root (two levels up from this file: src/data622/paths.py -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"             # Raw data - never edit raw data directly!
INTERIM_DATA_DIR = DATA_DIR / "interim"     # Interim data - data that is not raw, but also not yet model ready
PROCESSED_DATA_DIR = DATA_DIR / "processed" # Processed data - Model-ready data

# Other top-level directories
MODELS_DIR = PROJECT_ROOT / "models"         # Contains saved models
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"                  # Contains all notebooks
NOTEBOOKS_EXPLORATORY_DIR = NOTEBOOKS_DIR / "exploratory"  # EDA and experimentation notebooks
NOTEBOOKS_PRESENTATION_DIR = NOTEBOOKS_DIR / "presentation" # Polished, presentable notebooks
REFERENCES_DIR = PROJECT_ROOT / "references" # Contains reference materials like data dictionaries
SCRIPTS_DIR = PROJECT_ROOT / "scripts"       # Contains scripts to run full pipelines

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"         # Contains configuration files (e.g. model params, feature lists)

from src.data622.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REFERENCES_DIR,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "REFERENCES_DIR",
]
# Sample usage:
# from data622.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
#
# raw_file = RAW_DATA_DIR / "payroll.csv"
# processed_file = PROCESSED_DATA_DIR / "payroll_clean.parquet"
# model_file = MODELS_DIR / "model.pkl"
