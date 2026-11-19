"""Statistical tests and ranking metrics.

DeLong (fast, Sun & Xu 2014) for paired AUROCs, exact McNemar, Holm correction, rank-agreement
metrics (Spearman, Kendall, pairwise ranking accuracy, top-k overlap, MAE) and bootstrap
comparison of two dependent correlations.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------- DeLong

def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.stack([_midrank(pos[r]) for r in range(k)])
    ty = np.stack([_midrank(neg[r]) for r in range(k)])
    tz = np.stack([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Paired DeLong test of AUROC(p1) vs AUROC(p2) on the same cases."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    if m == 0 or m == len(y):
        return {"auc1": np.nan, "auc2": np.nan, "diff": np.nan, "z": np.nan, "p": np.nan}
    preds = np.vstack([np.asarray(p1, float)[order], np.asarray(p2, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z), "p": float(p)}


def delong_ci(y: np.ndarray, p: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    if m == 0 or m == len(y):
        return (np.nan, np.nan, np.nan)
    aucs, cov = _fast_delong(np.asarray(p, float)[order][None, :], m)
    se = np.sqrt(cov[0, 0])
    zq = stats.norm.ppf(1 - (1 - level) / 2)
    return float(aucs[0]), float(max(0, aucs[0] - zq * se)), float(min(1, aucs[0] + zq * se))


# ----------------------------------------------------------------------------- McNemar / Holm

def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    pv = p[ok][idx]
    m = len(pv)
    adj = np.maximum.accumulate([(m - i) * pv[i] for i in range(m)]) if m else np.array([])
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out.tolist()


# ----------------------------------------------------------------------------- ranking metrics

def _clean(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok]


def spearman(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic)


def kendall(x, y) -> float:
    x, y = _clean(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.kendalltau(x, y).statistic)


def pairwise_ranking_accuracy(pred, real) -> float:
    pred, real = _clean(pred, real)
    n = len(pred)
    if n < 2:
        return np.nan
    i, j = np.triu_indices(n, 1)
    dr = np.sign(real[i] - real[j])
    dp = np.sign(pred[i] - pred[j])
    keep = dr != 0
    if not keep.any():
        return np.nan
    return float(np.mean(dp[keep] == dr[keep]))


def topk_overlap(pred, real, k: int = 5) -> float:
    """Overlap of the k most robust configurations (smallest loss) by predicted and by real loss."""
    pred, real = _clean(pred, real)
    if len(pred) < k:
        return np.nan
    a = set(np.argsort(pred)[:k])
    b = set(np.argsort(real)[:k])
    return len(a & b) / k


def mae(pred, real) -> float:
    pred, real = _clean(pred, real)
    return float(np.mean(np.abs(pred - real))) if len(pred) else np.nan


RANK_METRICS = {"spearman": spearman, "kendall": kendall, "pra": pairwise_ranking_accuracy, "mae": mae}


def bootstrap_diff_p(diffs: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: difference = 0 (percentile method)."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return float(min(1.0, max(p, 1.0 / len(d))))


def zone(ci_low: float, ci_high: float, not_valid_upper: float = 0.5, valid_lower: float = 0.7) -> str:
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "undetermined"
    if ci_high < not_valid_upper:
        return "not valid"
    if ci_low > valid_lower:
        return "valid"
    return "inconclusive"
