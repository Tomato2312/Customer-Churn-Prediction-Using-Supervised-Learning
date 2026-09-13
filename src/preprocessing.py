"""Feature preprocessing: median/mode imputation, scaling, one-hot encoding."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return the ColumnTransformer used ahead of every model."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, config.NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, config.CATEGORICAL_FEATURES),
    ])


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable feature names after the ColumnTransformer, for SHAP plots."""
    numeric_names = list(config.NUMERIC_FEATURES)
    ohe = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_names = list(ohe.get_feature_names_out(config.CATEGORICAL_FEATURES))
    return numeric_names + categorical_names
