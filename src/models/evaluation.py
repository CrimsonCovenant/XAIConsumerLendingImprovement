"""Credit-scoring evaluation metrics and threshold selection.
Standard industry metrics for mortgage lending model assessment:
  AUC-ROC  -- how well the model ranks applicants (higher = better)
  KS stat  -- max separation between approval/denial distributions
  Gini     -- 2*AUC - 1 (industry convention, same info as AUC)
  F1       -- harmonic mean of precision and recall
Accuracy is reported only as a secondary check because it is misleading
on the imbalanced HMDA data (approvals outnumber denials ~3:1).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic -- standard credit-scoring metric.
    Measures the maximum gap between the cumulative distributions of
    approved and denied applications when sorted by predicted probability.
    A good scorecard targets KS >= 0.40.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    # Sort all predictions from lowest to highest probability
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    # Build running CDFs: what fraction of each class have we seen so far?
    cdf_pos = np.cumsum(y_sorted) / n_pos
    cdf_neg = np.cumsum(1 - y_sorted) / n_neg
    # KS is the biggest gap between the two CDFs
    return float(np.max(np.abs(cdf_pos - cdf_neg)))

def select_threshold_by_f1(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
    """Find the decision threshold that maximizes F1 score.
    We scan thresholds from 0.05 to 0.94 in steps of 0.01 instead of
    hardcoding 0.5, because the optimal cutoff depends on the class
    balance and cost tradeoff of the specific dataset.
    Returns (best_threshold, f1_at_best).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [
        f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])

def credit_scorecard_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label: str = "Test set",
    verbose: bool = True,
) -> dict:
    """Compute the full evaluation suite and optionally print a summary.
    Returns a dict with: auc, ks, gini, accuracy, f1, threshold.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    ks = ks_statistic(y_true, y_prob)
    gini = 2 * auc - 1
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if verbose:
        print(f"  {label}")
        print(f"  AUC-ROC  : {auc:.4f}  (rank-order quality, higher = better)")
        print(f"  KS stat  : {ks:.4f}  (separation power, target >= 0.40)")
        print(f"  Gini     : {gini:.4f}  (2*AUC - 1, industry standard)")
        print(f"  Accuracy : {acc:.4f}  (secondary, misleading on imbalanced data)")
        print(f"  F1 score : {f1:.4f}  (at threshold={threshold:.2f})")
        cm = confusion_matrix(y_true, y_pred)
        print(f"  Confusion matrix (rows=actual, cols=predicted):")
        print(f"    Deny  predicted: {cm[0, 0]:>8,} correct  {cm[0, 1]:>8,} wrong")
        print(f"    Appr  predicted: {cm[1, 0]:>8,} wrong    {cm[1, 1]:>8,} correct")
    return {
        "auc": round(auc, 4),
        "ks": round(ks, 4),
        "gini": round(gini, 4),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "threshold": round(threshold, 2),
    }
