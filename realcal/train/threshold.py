"""Frozen operating threshold: highest sensitivity subject to specificity >= target (on HQ validation)."""
from __future__ import annotations

import numpy as np


def binary_score(probs: np.ndarray, n_classes: int, negative_class: int = 0) -> np.ndarray:
    """Score used for thresholding: P(referable) for binary tasks, 1 - P(normal) for multi-class."""
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs
    if n_classes == 2:
        return probs[:, 1]
    return 1.0 - probs[:, negative_class]


def binary_target(labels: np.ndarray, n_classes: int, negative_class: int = 0) -> np.ndarray:
    labels = np.asarray(labels)
    return labels.astype(int) if n_classes == 2 else (labels != negative_class).astype(int)


def threshold_at_specificity(y: np.ndarray, score: np.ndarray, target: float = 0.90) -> dict:
    """Smallest threshold t (predict positive if score >= t) with specificity >= target."""
    y = np.asarray(y).astype(int)
    s = np.asarray(score, dtype=np.float64)
    neg = np.sort(s[y == 0])
    pos = s[y == 1]
    if len(neg) == 0:
        t = float(np.min(s)) if len(s) else 0.5
    else:
        k = int(np.ceil(target * len(neg)))
        k = min(max(k, 1), len(neg))
        t = float(np.nextafter(neg[k - 1], np.inf))
    spec = float(np.mean(neg < t)) if len(neg) else float("nan")
    sens = float(np.mean(pos >= t)) if len(pos) else float("nan")
    return {"threshold": t, "val_specificity": spec, "val_sensitivity": sens, "target_specificity": target,
            "n_val_pos": int(len(pos)), "n_val_neg": int(len(neg))}
