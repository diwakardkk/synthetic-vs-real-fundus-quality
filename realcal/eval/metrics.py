"""Image-level metrics: AUROC (binary / macro one-vs-rest), Se/Sp at a threshold, ECE, Brier."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Binary AUROC with p = positive score, or macro OvR AUROC with p = (N, K) probabilities."""
    y = np.asarray(y)
    p = np.asarray(p)
    ok = ~pd_isna(y)
    y, p = y[ok].astype(int), p[ok]
    if p.ndim == 1:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    K = p.shape[1]
    vals = []
    for k in range(K):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            vals.append(roc_auc_score(yk, p[:, k]))
    return float(np.mean(vals)) if vals else float("nan")


def pd_isna(a: np.ndarray) -> np.ndarray:
    import pandas as pd

    return pd.isna(a)


def sens_spec(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    se = float(np.mean(pred[y == 1] == 1)) if (y == 1).any() else float("nan")
    sp = float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else float("nan")
    return se, sp


def ece_equal_mass(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins (binary, p = P(y=1))."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    bins = np.array_split(order, min(n_bins, len(p)))
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += len(b) / len(p) * abs(p[b].mean() - y[b].mean())
    return float(e)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> list[tuple[float, float, int]]:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    order = np.argsort(p)
    return [(float(p[b].mean()), float(y[b].mean()), int(len(b))) for b in np.array_split(order, min(n_bins, max(len(p), 1))) if len(b)]


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)
