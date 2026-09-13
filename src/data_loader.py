"""Load the Telco Customer Churn dataset.

The original Colab notebook downloaded the dataset directly from Kaggle's
anonymous API endpoint at runtime. Kaggle now requires an authenticated
request for that download endpoint, so this module uses the official
`kaggle` CLI/API (with your Kaggle credentials) instead, and otherwise
just reads whatever `.xlsx`/`.csv` file is already sitting in `data/`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "ylchang/telco-customer-churn-1113"


def download_dataset(data_dir: Path = config.DATA_DIR) -> Path:
    """Download the dataset via the Kaggle API.

    Requires `pip install kaggle` and a valid `~/.kaggle/kaggle.json`
    (or KAGGLE_USERNAME / KAGGLE_KEY environment variables). Raises a
    clear error if credentials aren't configured, instead of failing
    inside a low-level HTTP call.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'kaggle' package is required to auto-download the dataset. "
            "Install it with `pip install kaggle`, place your API token at "
            "~/.kaggle/kaggle.json, and retry -- or manually download "
            f"'{KAGGLE_DATASET}' from Kaggle and place the .xlsx file at "
            f"{config.DATA_PATH}."
        ) from exc

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading %s from Kaggle...", KAGGLE_DATASET)
    api.dataset_download_files(KAGGLE_DATASET, path=str(data_dir), unzip=True)

    candidates = list(data_dir.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            f"Download completed but no .xlsx file was found in {data_dir}."
        )
    downloaded = candidates[0]
    if downloaded != config.DATA_PATH:
        downloaded.rename(config.DATA_PATH)
    return config.DATA_PATH


def load_data(data_path: Path = config.DATA_PATH, auto_download: bool = True) -> pd.DataFrame:
    """Load the raw Telco dataset, downloading it first if missing."""
    if not data_path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"{data_path} not found and auto_download=False. Place the "
                "dataset there manually or call download_dataset()."
            )
        download_dataset(data_path.parent)

    df = pd.read_excel(data_path)
    logger.info("Loaded %d customers x %d columns from %s", *df.shape, data_path)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop leakage/identifier columns and split into X, y."""
    X = df.drop(columns=[c for c in config.EXCLUDED_COLUMNS if c in df.columns] + [config.TARGET]).copy()
    y = df[config.TARGET].map({"No": 0, "Yes": 1})

    # 'Total Charges' is read as object dtype because of stray blank strings
    # for brand-new customers (Tenure Months == 0).
    if "Total Charges" in X.columns:
        X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")

    missing = [c for c in config.FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise KeyError(f"Expected feature columns missing from dataset: {missing}")

    return X[config.FEATURE_COLUMNS], y
