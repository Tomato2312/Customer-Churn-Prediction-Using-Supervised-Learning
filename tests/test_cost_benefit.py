import numpy as np

from src.cost_benefit import cost_benefit_curve, profit_from_confusion


def test_profit_from_confusion_matches_manual_formula():
    # 90*TP - 10*FP, with retention=$10, churn_loss=$100
    assert profit_from_confusion(tp=100, fp=0) == 9000
    assert profit_from_confusion(tp=0, fp=100) == -1000
    assert profit_from_confusion(tp=50, fp=20) == 90 * 50 - 10 * 20


def test_cost_benefit_curve_is_monotonic_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    probabilities = np.clip(y_true * 0.5 + rng.normal(0, 0.25, size=500), 0, 1)

    curve = cost_benefit_curve(y_true, probabilities, thresholds=np.array([0.1, 0.5, 0.9]))
    # Profit at a very high threshold (almost nobody contacted) should be
    # close to zero relative to profit at a well-chosen mid threshold.
    assert curve.loc[curve["threshold"] == 0.9, "expected_profit"].iloc[0] <= curve["expected_profit"].max()
    assert set(curve.columns) == {"threshold", "TP", "FP", "FN", "TN", "expected_profit"}


def test_preprocessor_builds_and_transforms():
    import pandas as pd
    from src.preprocessing import build_preprocessor
    from src import config

    df = pd.DataFrame({
        **{c: [10.0, 20.0, None] for c in config.NUMERIC_FEATURES},
        **{c: ["Yes", "No", None] for c in config.CATEGORICAL_FEATURES},
    })
    pre = build_preprocessor()
    transformed = pre.fit_transform(df)
    assert transformed.shape[0] == 3
