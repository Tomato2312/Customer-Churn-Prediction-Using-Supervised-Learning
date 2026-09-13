"""SHAP explainability for the trained churn pipeline (Step 2 of the roadmap).

Requires `pip install shap` and a fitted pipeline that exposes
`preprocessor` and `model` steps (see `train.py`). Not run as part of
`train.py` by default since SHAP on the full training set can be slow;
call `generate_summary_plot` explicitly, e.g. from a notebook or a
one-off `python -m src.explain` run once you have `models/churn_pipeline.joblib`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .preprocessing import get_output_feature_names


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    return transformed.toarray() if hasattr(transformed, "toarray") else transformed


def compute_shap_values(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """Return (shap_values, feature_names) for the churn (positive) class.

    Uses TreeExplainer when the final estimator is tree-based (Decision
    Tree / Random Forest, which is what the model-comparison step in this
    project selects) and falls back to a model-agnostic KernelExplainer
    (on a small background sample) otherwise, e.g. for Logistic Regression.
    """
    import shap

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_output_feature_names(preprocessor)

    X_background_t = _transform(pipeline, X_background)
    X_explain_t = _transform(pipeline, X_explain)

    model_type = type(model).__name__
    if model_type in {"RandomForestClassifier", "DecisionTreeClassifier"}:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(X_explain_t)
        # TreeExplainer on a binary sklearn classifier returns a list
        # [class_0_values, class_1_values]; keep the churn (class 1) view.
        shap_values = raw_values[1] if isinstance(raw_values, list) else raw_values
    else:
        background_sample = shap.sample(X_background_t, min(100, len(X_background_t)))
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        shap_values = explainer.shap_values(X_explain_t, nsamples=200)[1]

    return shap_values, feature_names, X_explain_t


def generate_summary_plot(
    model_path: Path = config.MODEL_PATH,
    output_path: Path = config.OUTPUT_DIR / "06_shap_summary.png",
    sample_size: int = 500,
):
    """Fit-free: loads the saved pipeline + a sample of X_test.csv (written
    by train.py) and saves a SHAP beeswarm summary plot."""
    import shap

    pipeline = joblib.load(model_path)
    X_test = pd.read_csv(config.OUTPUT_DIR / "X_test_sample.csv")
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=config.RANDOM_STATE)

    shap_values, feature_names, X_explain_t = compute_shap_values(pipeline, X_test, X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain_t, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {output_path}")


def waterfall_for_customer(pipeline, X_background: pd.DataFrame, customer_row: pd.DataFrame,
                            output_path: Path | None = None):
    """Single-customer SHAP waterfall, used by the Streamlit app."""
    import shap

    shap_values, feature_names, X_explain_t = compute_shap_values(
        pipeline, X_background, customer_row
    )
    model = pipeline.named_steps["model"]
    base_value = (
        model.predict_proba(_transform(pipeline, X_background))[:, 1].mean()
    )
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_explain_t[0],
        feature_names=feature_names,
    )
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation, show=False)
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return output_path
    return fig


if __name__ == "__main__":
    generate_summary_plot()
