"""Evaluation metrics for binary ADMET endpoints.

ROC-AUC is the headline because it is what the TDC leaderboard reports, but it
is reported alongside metrics that behave differently under class imbalance:

* **PR-AUC** (average precision) -- sensitive to performance on the minority
  class, where ROC-AUC can look flattering.
* **MCC** -- balanced correlation coefficient; near 0 for a model that only
  exploits the class prior, even when accuracy is high.
* **Balanced accuracy** -- mean of sensitivity and specificity.

Accuracy is included only for completeness. On BBB, predicting "crosses" for
every molecule already scores ~0.76 accuracy, which is why it is never the
headline number here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_FREE_METRICS = ("roc_auc", "pr_auc")


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary classification metrics from predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # A fold containing a single class makes AUC undefined rather than zero.
    single_class = len(np.unique(y_true)) < 2

    metrics = {
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float("nan")
        if single_class
        else float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    )
    return metrics


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability cutoff maximising F1.

    Must only ever be called on a validation split. Tuning the threshold on test
    data and then reporting test metrics is a subtle and common form of leakage.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Use the predicted probabilities themselves as candidate cutoffs. Rounding
    # them first can produce a cutoff strictly above every probability, which
    # silently flips the whole prediction to the negative class -- visible with
    # a constant-output model such as DummyClassifier.
    candidates = np.unique(y_prob)
    if candidates.size == 0:
        return 0.5
    if candidates.size > 200:
        candidates = np.unique(np.quantile(candidates, np.linspace(0.0, 1.0, 200)))

    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def aggregate(runs: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation across seeds, as ``<metric>_mean``/``_std``."""
    if not runs:
        return {}

    keys = [
        key
        for key in runs[0]
        if isinstance(runs[0][key], (int, float)) and not isinstance(runs[0][key], bool)
    ]
    summary: dict[str, float] = {"n_seeds": len(runs)}
    for key in keys:
        values = np.array([run[key] for run in runs], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return summary


def format_mean_std(summary: dict[str, float], metric: str, decimals: int = 3) -> str:
    """Render ``0.912 ± 0.008`` for tables."""
    mean = summary.get(f"{metric}_mean")
    std = summary.get(f"{metric}_std")
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
