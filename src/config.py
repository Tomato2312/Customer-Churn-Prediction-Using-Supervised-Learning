"""Central configuration for the churn pipeline.

Keeping every path, feature list, and business assumption in one place
means `train.py`, `predict.py`, `explain.py`, and the `app/` services all
stay in sync with a single source of truth.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "Telco_customer_churn.xlsx"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
OUTPUT_DIR = ROOT_DIR / "outputs"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
TARGET = "Churn Label"

# Dropped for label leakage or being non-predictive identifiers.
# NOTE vs. the original notebook: `City` and `Zip Code` are additionally
# excluded here. They carry ~1,129 and ~1,652 unique values respectively;
# one-hot encoding them blows up the feature space (>1,100 sparse columns)
# and invites overfitting to location noise rather than actual churn
# drivers. This is a deliberate "elevate the DS mindset" fix from the
# upgrade roadmap, not present in the original Colab notebook.
EXCLUDED_COLUMNS = [
    "CustomerID",
    "Count", "Country", "State",
    "Lat Long", "Latitude", "Longitude",
    "City", "Zip Code",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
]

NUMERIC_FEATURES = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns carried through into the retention priority list purely for
# business context (not used as model inputs beyond what's already in
# FEATURE_COLUMNS).
PRIORITY_LIST_COLUMNS = [
    "CustomerID", "Churn_probability", "Action", "Contract",
    "Internet Service", "Tenure Months", "Monthly Charges", "Payment Method",
]

# ---------------------------------------------------------------------------
# Business cost assumptions (Step 2: cost-benefit threshold selection)
# ---------------------------------------------------------------------------
# Cost of a retention offer/outreach extended to one customer.
RETENTION_COST = 10.0
# Estimated value lost when a customer churns (lost revenue / CLTV proxy).
CHURN_LOSS = 100.0
