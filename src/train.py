"""End-to-end training pipeline: EDA exports, model comparison, threshold
selection (F2 + cost-benefit), final test evaluation, and artifact export.

Usage:
    python -m src.train

Requires data/Telco_customer_churn.xlsx (see README "Setup" section).
Writes all tables/figures to outputs/ and the fitted pipeline to
models/churn_pipeline.joblib.
"""
from __future__ import annotations

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .cost_benefit import cost_benefit_curve
from .data_loader import load_data, split_features_target
from .preprocessing import build_preprocessor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")


def save_figure(filename: str):
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def calculate_metrics(y_true, probabilities, threshold=0.50) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision (churn)": precision_score(y_true, predictions, zero_division=0),
        "Recall (churn)": recall_score(y_true, predictions, zero_division=0),
        "F1 (churn)": f1_score(y_true, predictions, zero_division=0),
        "F2 (churn)": fbeta_score(y_true, predictions, beta=2, zero_division=0),
    }


def run_eda(df: pd.DataFrame):
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    overview = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_rate_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })
    overview.to_csv(config.OUTPUT_DIR / "data_overview.csv", encoding="utf-8-sig")

    target_rate = (
        df[config.TARGET].value_counts(dropna=False)
        .rename_axis(config.TARGET).reset_index(name="customer_count")
    )
    target_rate["rate_pct"] = (target_rate["customer_count"] / len(df) * 100).round(2)
    target_rate.to_csv(config.OUTPUT_DIR / "target_distribution.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for ax, feature, title in zip(
        axes,
        ["Contract", "Internet Service", "Payment Method"],
        ["Churn rate by contract type", "Churn rate by internet service", "Churn rate by payment method"],
    ):
        churn_rate = (
            pd.crosstab(df[feature], df[config.TARGET], normalize="index")
            .mul(100)["Yes"].sort_values(ascending=False)
        )
        sns.barplot(x=churn_rate.index, y=churn_rate.values, ax=ax, color="#d95f02")
        ax.set(title=title, xlabel=feature, ylabel="Churn rate (%)")
        ax.tick_params(axis="x", rotation=32)
        for position, value in enumerate(churn_rate.values):
            ax.text(position, value + 0.5, f"{value:.1f}%", ha="center", fontsize=9)
    save_figure("01_churn_rate_by_business_feature.png")


def compare_models(X_train, y_train, X_val, y_val, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1,
        ),
    }
    pipelines, rows = {}, []
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        rows.append({"model": name, **calculate_metrics(y_val, probabilities)})
        pipelines[name] = pipeline

    results = pd.DataFrame(rows).sort_values(["PR-AUC", "ROC-AUC"], ascending=False).reset_index(drop=True)
    results.to_csv(config.OUTPUT_DIR / "validation_model_comparison.csv", index=False, encoding="utf-8-sig")
    return results, pipelines, models


def select_threshold(best_model_name, pipelines, X_val, y_val):
    """F2-optimal threshold (recall-weighted) -- the primary selection rule."""
    probabilities = pipelines[best_model_name].predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    table = pd.DataFrame({
        "threshold": thresholds,
        "precision": [precision_score(y_val, probabilities >= t, zero_division=0) for t in thresholds],
        "recall": [recall_score(y_val, probabilities >= t, zero_division=0) for t in thresholds],
        "f1": [f1_score(y_val, probabilities >= t, zero_division=0) for t in thresholds],
        "f2": [fbeta_score(y_val, probabilities >= t, beta=2, zero_division=0) for t in thresholds],
    })
    best_threshold = table.loc[table["f2"].idxmax(), "threshold"]
    table.to_csv(config.OUTPUT_DIR / "threshold_selection.csv", index=False, encoding="utf-8-sig")
    return best_threshold, table, probabilities


def select_threshold_by_profit(y_val, probabilities):
    """Cost-benefit-optimal threshold (Step 2 of the roadmap)."""
    curve = cost_benefit_curve(y_val, probabilities)
    curve.to_csv(config.OUTPUT_DIR / "cost_benefit_validation.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 5))
    plt.plot(curve["threshold"], curve["expected_profit"])
    best_row = curve.loc[curve["expected_profit"].idxmax()]
    plt.axvline(best_row["threshold"], color="black", linestyle="--",
                label=f"Profit-optimal threshold: {best_row['threshold']:.2f}")
    plt.title(f"Expected profit by threshold (retention=${config.RETENTION_COST:.0f}, churn loss=${config.CHURN_LOSS:.0f})")
    plt.xlabel("Churn probability threshold")
    plt.ylabel("Expected profit ($)")
    plt.legend()
    save_figure("06_cost_benefit_threshold.png")
    return best_row["threshold"], curve


def main():
    df = load_data()
    run_eda(df)
    X, y = split_features_target(df)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=config.RANDOM_STATE
    )
    pd.DataFrame([
        ["Train", len(y_train), y_train.mean()],
        ["Validation", len(y_val), y_val.mean()],
        ["Test", len(y_test), y_test.mean()],
    ], columns=["split", "customers", "churn_rate"]).to_csv(
        config.OUTPUT_DIR / "split_summary.csv", index=False, encoding="utf-8-sig"
    )

    preprocessor = build_preprocessor()
    validation_results, pipelines, models = compare_models(X_train, y_train, X_val, y_val, preprocessor)
    best_model_name = validation_results.loc[0, "model"]

    f2_threshold, threshold_table, val_probabilities = select_threshold(best_model_name, pipelines, X_val, y_val)
    profit_threshold, profit_curve = select_threshold_by_profit(y_val, val_probabilities)

    print(f"Best model: {best_model_name}")
    print(f"F2-optimal threshold: {f2_threshold:.2f}")
    print(f"Profit-optimal threshold: {profit_threshold:.2f}  "
          f"(expected profit ${profit_curve['expected_profit'].max():,.0f} on validation)")

    # Final fit on train+val, evaluated once on the untouched test set,
    # using the F2 threshold as the production default (see README for
    # the trade-off discussion between the two thresholds).
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", models[best_model_name])])
    final_pipeline.fit(X_train_val, y_train_val)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= f2_threshold).astype(int)

    test_metrics = pd.DataFrame([calculate_metrics(y_test, test_probabilities, f2_threshold)], index=[best_model_name])
    test_metrics.to_csv(config.OUTPUT_DIR / "test_metrics.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, test_predictions),
                            display_labels=["Stay", "Churn"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix")
    RocCurveDisplay.from_predictions(y_test, test_probabilities, ax=axes[1])
    axes[1].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, test_probabilities, ax=axes[2])
    axes[2].set_title("Precision-Recall curve")
    save_figure("05_test_evaluation.png")

    report = classification_report(y_test, test_predictions, target_names=["Stay", "Churn"], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(config.OUTPUT_DIR / "classification_report.csv", encoding="utf-8-sig")

    retention_list = X_test.copy()
    retention_list.insert(0, "CustomerID", df.loc[X_test.index, "CustomerID"])
    retention_list["Churn_probability"] = test_probabilities
    retention_list["Action"] = np.where(test_predictions == 1, "Priority retention contact", "Monitor")
    retention_list[config.PRIORITY_LIST_COLUMNS].sort_values(
        "Churn_probability", ascending=False
    ).to_csv(config.OUTPUT_DIR / "retention_priority_list.csv", index=False, encoding="utf-8-sig")

    # Persist artifacts for app/, predict.py, and explain.py
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, config.MODEL_PATH)
    X_test.to_csv(config.OUTPUT_DIR / "X_test_sample.csv", index=False)

    print(f"\nArtifacts written to {config.OUTPUT_DIR} and {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
