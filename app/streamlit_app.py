"""Interactive churn-scoring app.

Run with:
    streamlit run app/streamlit_app.py

Requires models/churn_pipeline.joblib (produced by `python -m src.train`)
and, for the SHAP waterfall, `pip install shap` plus
outputs/X_test_sample.csv (also written by train.py, used as a background
distribution for the explainer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.explain import waterfall_for_customer  # noqa: E402

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_pipeline():
    if not config.MODEL_PATH.exists():
        st.error(
            f"No trained model found at `{config.MODEL_PATH}`. "
            "Run `python -m src.train` first (requires data/Telco_customer_churn.xlsx)."
        )
        st.stop()
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_background():
    path = config.OUTPUT_DIR / "X_test_sample.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


CATEGORY_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["Yes", "No"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "Phone Service": ["Yes", "No"],
    "Multiple Lines": ["Yes", "No", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["Yes", "No", "No internet service"],
    "Online Backup": ["Yes", "No", "No internet service"],
    "Device Protection": ["Yes", "No", "No internet service"],
    "Tech Support": ["Yes", "No", "No internet service"],
    "Streaming TV": ["Yes", "No", "No internet service"],
    "Streaming Movies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["Yes", "No"],
    "Payment Method": [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}


def sidebar_inputs() -> pd.DataFrame:
    st.sidebar.header("Customer profile")
    values = {}
    values["Tenure Months"] = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    values["Monthly Charges"] = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
    values["Total Charges"] = st.sidebar.number_input(
        "Total charges ($)", 0.0, 9000.0, float(values["Monthly Charges"] * values["Tenure Months"])
    )
    values["CLTV"] = st.sidebar.slider("Customer lifetime value (CLTV score)", 2000, 6500, 4000)

    st.sidebar.divider()
    for col, options in CATEGORY_OPTIONS.items():
        values[col] = st.sidebar.selectbox(col, options)

    return pd.DataFrame([values])[config.FEATURE_COLUMNS]


def main():
    st.title("📉 Customer Churn Predictor")
    st.caption(
        "Enter a customer profile in the sidebar to get a churn probability, "
        "a recommended action, and (if SHAP is installed) the top drivers "
        "behind that specific prediction."
    )

    pipeline = load_pipeline()
    background = load_background()
    customer = sidebar_inputs()

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.37, 0.01,
        help="Default 0.37 is the F2-optimal threshold from validation (see README).",
    )

    probability = pipeline.predict_proba(customer)[0, 1]
    action = "🔴 Priority retention contact" if probability >= threshold else "🟢 Monitor"

    col1, col2 = st.columns(2)
    col1.metric("Churn probability", f"{probability:.1%}")
    col2.metric("Recommended action", action)

    st.divider()
    st.subheader("Why this prediction?")
    if background is None:
        st.info(
            "No background sample found at outputs/X_test_sample.csv. "
            "Run `python -m src.train` to generate one for SHAP explanations."
        )
        return

    try:
        import shap  # noqa: F401
    except ImportError:
        st.info("Install SHAP (`pip install shap`) to see a per-customer feature-attribution waterfall.")
        return

    with st.spinner("Computing SHAP attribution..."):
        fig = waterfall_for_customer(pipeline, background, customer)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
