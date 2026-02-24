# ml/evaluate.py
"""
Metric utilities for credit risk modeling (Phase 2+).

- Keep train.py clean.
- Reuse the same evaluation metrics across Logistic Regression baseline
  and later XGBoost/LightGBM models.

Included metrics:
- ROC-AUC
- PR-AUC (Average Precision)
- KS statistic (classic credit scoring metric)
- Recall@FixedFPR (optional, good for policy constraints)

All functions expect:
- y_true in {0, 1}
- y_score as probability/PD for class 1
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute KS statistic:
    KS = max_t | CDF_good(t) - CDF_bad(t) |

    Implementation detail:
    - Sort scores ascending.
    - Build cumulative distributions of good vs bad.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    order = np.argsort(y_score)
    y_sorted = y_true[order]

    bad = (y_sorted == 1).astype(float)
    good = (y_sorted == 0).astype(float)

    bad_cumulative = np.cumsum(bad) / (bad.sum() + 1e-12)
    good_cumulative = np.cumsum(good) / (good.sum() + 1e-12)

    return float(np.max(np.abs(good_cumulative - bad_cumulative)))


def recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    Recall (TPR) at a specified false positive rate.

    Useful when operations/policy requires:
    - "keep false positives under X%, maximize recall"

    If ROC curve doesn't hit exactly target_fpr, pick closest point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmin(np.abs(fpr - target_fpr)))
    return float(tpr[idx])


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fixed_fpr: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute a standard metric bundle for PD modeling.

    Args:
      y_true: array-like, 0/1
      y_score: predicted probability/PD for class 1
      fixed_fpr: if provided, compute recall_at_fixed_fpr

    Returns:
      dict with roc_auc, pr_auc, ks, and optionally recall_at_fpr_xxx
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    out: Dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "ks": float(ks_statistic(y_true, y_score)),
    }

    if fixed_fpr is not None:
        out[f"recall_at_fpr_{fixed_fpr:.3f}"] = recall_at_fixed_fpr(y_true, y_score, fixed_fpr)

    return out