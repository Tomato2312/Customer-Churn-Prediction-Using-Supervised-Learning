"""Score new customers with the saved pipeline.

Usage:
    python -m src.predict --input new_customers.csv --output scored.csv \
        --threshold 0.37

`new_customers.csv` must contain at least the columns in
`config.FEATURE_COLUMNS` (extra columns, e.g. CustomerID, are passed
through untouched).
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config


def load_pipeline(model_path=config.MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def score(df: pd.DataFrame, pipeline, threshold: float = 0.37) -> pd.DataFrame:
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")

    X = df[config.FEATURE_COLUMNS].copy()
    if "Total Charges" in X.columns:
        X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")

    probabilities = pipeline.predict_proba(X)[:, 1]
    result = df.copy()
    result["Churn_probability"] = probabilities
    result["Action"] = np.where(
        probabilities >= threshold, "Priority retention contact", "Monitor"
    )
    return result.sort_values("Churn_probability", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of new customers to score")
    parser.add_argument("--output", required=True, help="Where to write the scored CSV")
    parser.add_argument("--threshold", type=float, default=0.37,
                         help="Churn-probability threshold for flagging priority contact")
    args = parser.parse_args()

    pipeline = load_pipeline()
    df = pd.read_csv(args.input)
    scored = score(df, pipeline, args.threshold)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored)} customers -> {args.output}")
    print(scored.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
