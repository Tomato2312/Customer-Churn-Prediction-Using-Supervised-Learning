"""Cost-benefit threshold selection (Step 2 of the upgrade roadmap).

Instead of (or alongside) the F2-optimal threshold, this picks the
threshold that maximizes *expected business profit* under a simple
retention-campaign cost model:

  - Contacting a customer (predicted positive) costs `RETENTION_COST`,
    regardless of whether they were actually going to churn.
  - A successfully retained true churner (TP) avoids a `CHURN_LOSS` of
    lost revenue/CLTV, net of the retention cost already paid.
  - A churner who is *not* contacted (FN) is assumed lost: full
    `CHURN_LOSS`.
  - A loyal customer who is *not* contacted (TN) costs nothing.

  Expected profit(t) = (CHURN_LOSS - RETENTION_COST) * TP(t)
                        - RETENTION_COST * FP(t)

This is a simplification (it assumes retention offers are 100%
effective and CLTV is uniform across customers) but it is the standard
starting point for a churn cost-benefit analysis and is enough to show
*why* a given threshold is preferable to the naive default of 0.50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import config


def profit_from_confusion(tp: float, fp: float, retention_cost: float = config.RETENTION_COST,
                           churn_loss: float = config.CHURN_LOSS) -> float:
    benefit_per_tp = churn_loss - retention_cost
    return benefit_per_tp * tp - retention_cost * fp


def cost_benefit_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
    retention_cost: float = config.RETENTION_COST,
    churn_loss: float = config.CHURN_LOSS,
) -> pd.DataFrame:
    """Expected profit at each threshold, computed directly from predictions."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    rows = []
    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        profit = profit_from_confusion(tp, fp, retention_cost, churn_loss)
        rows.append({
            "threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "expected_profit": profit,
        })
    return pd.DataFrame(rows)


def cost_benefit_curve_from_threshold_table(
    threshold_table: pd.DataFrame,
    n_positive: int,
    n_negative: int,
    retention_cost: float = config.RETENTION_COST,
    churn_loss: float = config.CHURN_LOSS,
) -> pd.DataFrame:
    """Same idea, but reconstructed from a precision/recall-per-threshold
    table (i.e. `outputs/threshold_selection.csv`) when raw predictions
    aren't available. TP is derived from recall, FP from precision + TP.
    """
    rows = []
    for _, r in threshold_table.iterrows():
        precision, recall = r["precision"], r["recall"]
        tp = recall * n_positive
        fp = tp * (1 - precision) / precision if precision > 0 else 0.0
        fn = n_positive - tp
        profit = profit_from_confusion(tp, fp, retention_cost, churn_loss)
        rows.append({
            "threshold": r["threshold"], "TP": tp, "FP": fp, "FN": fn,
            "expected_profit": profit,
        })
    return pd.DataFrame(rows)


def best_threshold_by_profit(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["expected_profit"].idxmax()]
