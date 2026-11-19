"""Experiment 3: RealCal calibration.

* Level 1a (primary): non-negative weights on 6 group-level synthetic losses, L1 penalty, lambda by
  nested leave-one-backbone-out (LOBO) CV.
* Level 1b: the same over 75 conditions; selection frequency over backbone bootstrap resamples.
* Level 2: simplex mixture pi of corruption conditions minimising MMD^2 between real-LQ features and
  the corrupted-HQ mixture (frozen analysis encoder, Gaussian kernel, median heuristic).
* Realism: classifier two-sample test (real LQ vs synthetic) for uniform, Level-1 and Level-2 mixes.
* Baselines B1-B5; LOBO (primary) and leave-one-family-out (supplement) validation.
* Bootstrap CIs (two-level), bootstrap difference tests vs B1, Holm correction.
* Writes ``prereg/frozen_weights.json``.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from realcal.config import Ctx, config_hash
from realcal.corruptions.registry import by_name
from realcal.experiments.common import (Analysis, build_analysis, config_frame, delta_real, delta_syn,
                                        group_scores, rank_stats, strip_boot)
from realcal.stats.bootstrap import config_index_for_backbones, percentile_ci
from realcal.stats.tests import RANK_METRICS, bootstrap_diff_p, holm
from realcal.utils.io import read_json, write_csv, write_json
from realcal.utils.provenance import code_hash

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- solvers

def fit_nn_l1(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """min_{b, w>=0} ||y - b - Xw||^2 + lam * sum(w). Returns (b, w)."""
    n, p = X.shape
    try:
        import cvxpy as cp

        w = cp.Variable(p, nonneg=True)
        b = cp.Variable()
        obj = cp.Minimize(cp.sum_squares(y - b - X @ w) + lam * cp.sum(w))
        prob = cp.Problem(obj)
        prob.solve(solver=cp.CLARABEL) if "CLARABEL" in cp.installed_solvers() else prob.solve()
        if w.value is None:
            raise RuntimeError(prob.status)
        return float(b.value), np.maximum(np.asarray(w.value).ravel(), 0.0)
    except Exception as e:  # noqa: BLE001
        log.debug("cvxpy failed (%s); using L-BFGS-B", e)
        from scipy.optimize import minimize

        def f(z):
            r = y - z[0] - X @ z[1:]
            return float(r @ r + lam * z[1:].sum()), np.concatenate([[-2 * r.sum()], -2 * X.T @ r + lam])

        z0 = np.concatenate([[y.mean()], np.zeros(p)])
        res = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + [(0, None)] * p)
        return float(res.x[0]), np.maximum(res.x[1:], 0)


def lambda_grid(ctx: Ctx) -> list[float]:
    c = ctx.cfg.calibration
    return [0.0] + list(np.logspace(np.log10(float(c.lambda_grid_min)), np.log10(float(c.lambda_grid_max)), int(c.lambda_grid_n)))


def _lobo_preds(X, y, groups_, lam) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups_):
        tr, te = groups_ != g, groups_ == g
        if tr.sum() < 2:
            continue
        b, w = fit_nn_l1(X[tr], y[tr], lam)
        pred[te] = b + X[te] @ w
    return pred


def select_lambda(X, y, groups_, grid) -> float:
    if len(np.unique(groups_)) < 3:
        return float(grid[0])
    mses = []
    for lam in grid:
        p = _lobo_preds(X, y, groups_, lam)
        ok = ~np.isnan(p)
        mses.append(np.mean((p[ok] - y[ok]) ** 2) if ok.any() else np.inf)
    mses = np.asarray(mses)
    best = np.flatnonzero(mses <= mses.min() + 1e-12)
    return float(grid[best[-1]])  # ties -> larger penalty


@dataclass
class FoldModel:
    held_out: str
    b: float
    w: np.ndarray
    lam: float


def nested_lobo(X, y, groups_, grid) -> tuple[np.ndarray, list[FoldModel]]:
    """Outer LOBO with lambda chosen by inner LOBO on the training backbones."""
    pred = np.full(len(y), np.nan)
    models = []
    for g in np.unique(groups_):
        tr, te = groups_ != g, groups_ == g
        if tr.sum() < 2:
            continue
        lam = select_lambda(X[tr], y[tr], groups_[tr], grid)
        b, w = fit_nn_l1(X[tr], y[tr], lam)
        pred[te] = b + X[te] @ w
        models.append(FoldModel(str(g), b, w, lam))
    return pred, models


def apply_fold_models(models: list[FoldModel], X: np.ndarray, groups_: np.ndarray) -> np.ndarray:
    """Predictions with each configuration scored by the model that held out its group."""
    out = np.full(X.shape[:-1], np.nan)
    for m in models:
        sel = groups_ == m.held_out
        out[..., sel] = m.b + X[..., sel, :] @ m.w
    return out


REL_GRID = [0.0] + list(np.logspace(-3, 0, 16))  # penalty as a fraction of lambda_max (standardised variant)


@dataclass
class StdFoldModel:
    held_out: str
    b: float
    w: np.ndarray
    lam: float
    mu: np.ndarray
    sd: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.b + ((X - self.mu) / self.sd) @ self.w


def fit_std(X: np.ndarray, y: np.ndarray, rel_lam: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Sensitivity variant: features z-scored, penalty relative to lambda_max (smallest penalty that
    zeroes every non-negative weight), as in glmnet. Returns (b, w on the z scale, mu, sd)."""
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    lam_max = float(max(2 * np.max(Z.T @ (y - y.mean())), 1e-12))
    b, w = fit_nn_l1(Z, y, rel_lam * lam_max)
    return b, w, mu, sd


def _lobo_preds_std(X, y, groups_, rel_lam) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups_):
        tr, te = groups_ != g, groups_ == g
        if tr.sum() < 3:
            continue
        b, w, mu, sd = fit_std(X[tr], y[tr], rel_lam)
        pred[te] = b + ((X[te] - mu) / sd) @ w
    return pred


def select_lambda_std(X, y, groups_, grid=REL_GRID) -> float:
    if len(np.unique(groups_)) < 3:
        return float(grid[0])
    mses = []
    for lam in grid:
        p = _lobo_preds_std(X, y, groups_, lam)
        ok = ~np.isnan(p)
        mses.append(np.mean((p[ok] - y[ok]) ** 2) if ok.any() else np.inf)
    mses = np.asarray(mses)
    best = np.flatnonzero(mses <= mses.min() + 1e-12)
    return float(grid[best[-1]])


def nested_lobo_std(X, y, groups_, grid=REL_GRID) -> tuple[np.ndarray, list[StdFoldModel]]:
    pred = np.full(len(y), np.nan)
    models = []
    for g in np.unique(groups_):
        tr, te = groups_ != g, groups_ == g
        if tr.sum() < 3:
            continue
        lam = select_lambda_std(X[tr], y[tr], groups_[tr], grid)
        b, w, mu, sd = fit_std(X[tr], y[tr], lam)
        m = StdFoldModel(str(g), b, w, lam, mu, sd)
        pred[te] = m.predict(X[te])
        models.append(m)
    return pred, models


def apply_std_models(models: list[StdFoldModel], X: np.ndarray, groups_: np.ndarray) -> np.ndarray:
    out = np.full(X.shape[:-1], np.nan)
    for m in models:
        sel = groups_ == m.held_out
        out[..., sel] = m.b + ((X[..., sel, :] - m.mu) / m.sd) @ m.w
    return out


def n_intercept_only(models) -> int:
    """Number of LOBO folds whose fitted model has every weight at zero (prediction = training mean)."""
    return int(sum(bool(np.all(np.asarray(m.w) <= 1e-8)) for m in models))


RANK_COLS = ["spearman", "spearman_ci", "kendall", "kendall_ci", "pra", "pra_ci", "top5", "diff_vs_B1",
             "diff_vs_B1_lo", "diff_vs_B1_hi", "p_vs_B1"]


def mask_intercept_only(table: pd.DataFrame, fold_models: dict[str, list]) -> pd.DataFrame:
    """Ranking statistics of a method whose every LOBO fold is intercept-only are an artefact (the
    held-out prediction is the mean of the *other* backbones, which is negatively related to the
    held-out loss by construction). They are reported as NA; MAE is kept."""
    t = table.copy().astype({c: object for c in RANK_COLS if c in table})
    t["intercept_only_folds"] = ""
    for name, models in fold_models.items():
        sel = t["method"] == name
        if not sel.any() or not models:
            continue
        k = n_intercept_only(models)
        t.loc[sel, "intercept_only_folds"] = f"{k}/{len(models)}"
        if k == len(models):
            for c in RANK_COLS:
                if c in t:
                    t.loc[sel, c] = np.nan
            log.warning("%s: intercept-only in every LOBO fold -> ranking statistics reported as NA", name)
        elif k > 0:
            log.warning("%s: intercept-only in %d of %d LOBO folds (ranking statistics partly artefactual)",
                        name, k, len(models))
    if "p_vs_B1" in t:
        t["p_vs_B1_holm"] = holm(pd.to_numeric(t["p_vs_B1"], errors="coerce").tolist())
    return t


def lobo_linear(score: np.ndarray, y: np.ndarray, groups_: np.ndarray) -> tuple[np.ndarray, dict]:
    """LOBO univariate linear recalibration of a raw score (for MAE on the loss scale)."""
    pred = np.full(len(y), np.nan)
    coefs = {}
    for g in np.unique(groups_):
        tr, te = (groups_ != g) & ~np.isnan(score) & ~np.isnan(y), groups_ == g
        if tr.sum() < 2 or np.ptp(score[tr]) == 0:
            continue
        a, c = np.polyfit(score[tr], y[tr], 1)
        pred[te] = a * score[te] + c
        coefs[str(g)] = (a, c)
    return pred, coefs


# ----------------------------------------------------------------------------- Level 2

def _pca(Z: np.ndarray, k: int = 256) -> Callable[[np.ndarray], np.ndarray]:
    mu = Z.mean(0)
    sd = Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    k = min(k, Zs.shape[0] - 1, Zs.shape[1])
    _, _, Vt = np.linalg.svd(Zs, full_matrices=False)
    V = Vt[:k].T
    return lambda A: ((A - mu) / sd) @ V


def _sqd(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0)


def mmd2_unbiased(A: np.ndarray, B: np.ndarray, gamma: float) -> float:
    kaa = np.exp(-gamma * _sqd(A, A))
    kbb = np.exp(-gamma * _sqd(B, B))
    kab = np.exp(-gamma * _sqd(A, B))
    n, m = len(A), len(B)
    if n < 2 or m < 2:
        return np.nan
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())


def level2_pi(ctx: Ctx, real: np.ndarray, synth: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Simplex weights pi minimising MMD^2(real, sum_k pi_k synth_k)."""
    import cvxpy as cp

    names = list(synth)
    rng = np.random.default_rng(0)
    pooled = np.concatenate([real] + [synth[k] for k in names])
    sub = pooled[rng.choice(len(pooled), size=min(2000, len(pooled)), replace=False)]
    proj = _pca(sub)
    R = proj(real)
    S = [proj(synth[k]) for k in names]
    ps = proj(sub)
    d = _sqd(ps[:500], ps[:500])
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    gamma = 1.0 / med
    K = len(names)
    G = np.zeros((K, K))
    c = np.zeros(K)
    for i in range(K):
        c[i] = np.exp(-gamma * _sqd(R, S[i])).mean()
        for j in range(i, K):
            G[i, j] = G[j, i] = np.exp(-gamma * _sqd(S[i], S[j])).mean()
    G = (G + G.T) / 2 + 1e-8 * np.eye(K)
    pi = cp.Variable(K, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.quad_form(pi, cp.psd_wrap(G)) - 2 * c @ pi), [cp.sum(pi) == 1])
    prob.solve()
    p = np.maximum(np.asarray(pi.value).ravel(), 0)
    p = p / p.sum()
    rr = np.exp(-gamma * _sqd(R, R)).mean()
    info = {"gamma": float(gamma), "mmd2_pi": float(p @ G @ p - 2 * c @ p + rr),
            "mmd2_uniform": float(np.full(K, 1 / K) @ G @ np.full(K, 1 / K) - 2 * c.mean() + rr), "proj": proj}
    return p, info


# ----------------------------------------------------------------------------- realism test

def realism_test(ctx: Ctx, real: np.ndarray, real_groups: np.ndarray, synth: dict[str, np.ndarray],
                 synth_groups: dict[str, np.ndarray], mix: dict[str, float], seed: int, proj, gamma: float) -> dict:
    """Classifier two-sample test real LQ vs synthetic mixture; grouped 5-fold CV by patient."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    names = [k for k in mix if mix[k] > 0 and k in synth]
    probs = np.array([mix[k] for k in names], float)
    probs /= probs.sum()
    n = len(real)
    pick = rng.choice(len(names), size=n, p=probs)
    S, Sg = [], []
    for k_i in range(len(names)):
        cnt = int((pick == k_i).sum())
        if cnt == 0:
            continue
        arr = synth[names[k_i]]
        idx = rng.choice(len(arr), size=cnt, replace=len(arr) < cnt)
        S.append(arr[idx])
        Sg.append(synth_groups[names[k_i]][idx])
    S = np.concatenate(S)
    Sg = np.concatenate(Sg)
    X = np.concatenate([real, S])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(S))])
    g = np.concatenate([real_groups, Sg])
    oof = np.zeros(len(y))
    nf = min(int(ctx.cfg.calibration.realism_folds), len(np.unique(g)))
    for tr, te in GroupKFold(n_splits=nf).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    from realcal.eval.metrics import auroc

    a = auroc(y, oof)
    boots = []
    ug, inv = np.unique(g, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(ug))]
    for b in range(min(int(ctx.cfg.stats.n_boot), 1000)):
        r2 = np.random.default_rng(seed * 31 + b)
        idx = np.concatenate([rows_of[i] for i in r2.integers(0, len(ug), len(ug))])
        boots.append(auroc(y[idx], oof[idx]))
    lo, hi = percentile_ci(np.asarray(boots), float(ctx.cfg.stats.ci))
    mmd = mmd2_unbiased(proj(real), proj(S), gamma)
    return {"auroc": a, "auroc_ci": [lo, hi], "mmd2": mmd, "n_real": int(len(real)), "n_synth": int(len(S))}


def load_phi(ctx: Ctx, dataset: str, conditions: list[str]) -> tuple[np.ndarray, np.ndarray, dict, dict, list[str]]:
    """Analysis-encoder features: real LQ images and corrupted HQ images per condition (subsampled)."""
    from realcal.corruptions.cache import eval_hq_ids
    from realcal.data.common import load_metadata
    from realcal.models.features import get_features

    enc = str(ctx.cfg.features.level2_encoder)
    meta = load_metadata(ctx, dataset)
    mx = int(ctx.cfg.calibration.mmd_max_per_set)
    rng = np.random.default_rng(0)
    lq = meta[meta["lq_any"]]
    real_ids = lq["image_id"].tolist()
    real = get_features(ctx, enc, dataset, 224, "clean", real_ids)
    real_g = lq["patient_id"].astype(str).to_numpy()
    hq_ids = eval_hq_ids(ctx, dataset)
    pmap = dict(zip(meta["image_id"], meta["patient_id"].astype(str)))
    sel = sorted(rng.choice(len(hq_ids), size=min(mx, len(hq_ids)), replace=False).tolist())
    ids = [hq_ids[i] for i in sel]
    synth, sg = {}, {}
    for c in conditions:
        synth[c] = get_features(ctx, enc, dataset, 224, c, ids)
        sg[c] = np.array([pmap[i] for i in ids])
    return real, real_g, synth, sg, ids


# ----------------------------------------------------------------------------- main

def _methods_table(ctx: Ctx, an: Analysis, y: np.ndarray, yb: np.ndarray | None, preds: dict[str, np.ndarray],
                   preds_b: dict[str, np.ndarray | None], ref: str = "B1_uniform") -> pd.DataFrame:
    lvl = float(ctx.cfg.stats.ci)
    rows = []
    for name, p in preds.items():
        r = rank_stats(an, p, y, preds_b.get(name), yb, lvl)
        rows.append({"method": name, **strip_boot(r), "_boot": {k: r.get(f"{k}_boot") for k in RANK_METRICS}})
    t = pd.DataFrame(rows)
    refrow = t[t["method"] == ref].iloc[0] if (t["method"] == ref).any() else None
    pvals = []
    for i, r in t.iterrows():
        if refrow is None or r["method"] == ref or r["_boot"].get("spearman") is None or refrow["_boot"].get("spearman") is None:
            t.loc[i, "diff_vs_B1"] = np.nan
            pvals.append(np.nan)
            continue
        d = np.asarray(r["_boot"]["spearman"]) - np.asarray(refrow["_boot"]["spearman"])
        t.loc[i, "diff_vs_B1"] = r["spearman"] - refrow["spearman"]
        lo, hi = percentile_ci(d, lvl)
        t.loc[i, "diff_vs_B1_lo"], t.loc[i, "diff_vs_B1_hi"] = lo, hi
        pvals.append(bootstrap_diff_p(d))
    t["p_vs_B1"] = pvals
    t["p_vs_B1_holm"] = holm(pvals)
    return t.drop(columns=["_boot"])


def run_exp3(ctx: Ctx) -> dict:
    ds_name = "deepdrid"
    an = build_analysis(ctx, ds_name)
    if an is None:
        raise RuntimeError("no predictions; run the DeepDRiD chain first")
    lookup = by_name(ctx)
    bb = np.array([c.split("/")[0] for c in an.configs])
    fam = np.array([ctx.family(b) for b in bb])
    y = delta_real(an, "any")
    yb = delta_real(an, "any", boot=True) if an.boot is not None else None
    D = delta_syn(an)
    Db = delta_syn(an, boot=True) if an.boot is not None else None
    G, gnames = group_scores(ctx, an, D)
    Gb = group_scores(ctx, an, Db)[0] if Db is not None else None
    okc = ~np.isnan(y)
    gcols = ~np.isnan(G[okc]).any(0)
    ccols = ~np.isnan(D[okc]).any(0)
    G, gnames = G[:, gcols], [g for g, k in zip(gnames, gcols) if k]
    Gb = Gb[..., gcols] if Gb is not None else None
    conds = [c for c, k in zip(an.syn_conditions, ccols) if k]
    D, Db = D[:, ccols], (Db[..., ccols] if Db is not None else None)
    grid = lambda_grid(ctx)

    # Level 1a ----------------------------------------------------------------
    p1a, m1a = nested_lobo(G[okc], y[okc], bb[okc], grid)
    pred_1a = np.full(len(y), np.nan)
    pred_1a[okc] = p1a
    lam1a = select_lambda(G[okc], y[okc], bb[okc], grid)
    b1a, w1a = fit_nn_l1(G[okc], y[okc], lam1a)
    # Level 1a, standardised-penalty sensitivity variant -------------------------
    p1s, m1s = nested_lobo_std(G[okc], y[okc], bb[okc])
    pred_1s = np.full(len(y), np.nan)
    pred_1s[okc] = p1s
    lam1s = select_lambda_std(G[okc], y[okc], bb[okc])
    b1s, w1s, mu1s, sd1s = fit_std(G[okc], y[okc], lam1s)
    # Level 1b ----------------------------------------------------------------
    p1b, m1b = nested_lobo(D[okc], y[okc], bb[okc], grid)
    pred_1b = np.full(len(y), np.nan)
    pred_1b[okc] = p1b
    lam1b = select_lambda(D[okc], y[okc], bb[okc], grid)
    int1b, w1b = fit_nn_l1(D[okc], y[okc], lam1b)
    # LOFO (supplement) -------------------------------------------------------
    p_lofo, m_lofo = nested_lobo(G[okc], y[okc], fam[okc], grid)
    pred_lofo = np.full(len(y), np.nan)
    pred_lofo[okc] = p_lofo

    # bootstrap of weights (refit on two-level replicates with fixed lambda) --
    nrep = min(int(ctx.cfg.calibration.level1b_bootstrap), an.boot.shape[0] if an.boot is not None else 0)
    w1a_reps, w1b_reps = [], []
    for r in range(nrep):
        idx = config_index_for_backbones(an.engine, an.boot_bb[r])
        idx = idx[~np.isnan(yb[r, idx])]
        Xa = Gb[r, idx]
        Xb = Db[r, idx]
        keep = ~np.isnan(Xa).any(1) & ~np.isnan(Xb).any(1)
        if keep.sum() < 3:
            continue
        w1a_reps.append(fit_nn_l1(Xa[keep], yb[r, idx][keep], lam1a)[1])
        w1b_reps.append(fit_nn_l1(Xb[keep], yb[r, idx][keep], lam1b)[1])
    w1a_reps = np.asarray(w1a_reps) if w1a_reps else np.zeros((0, len(gnames)))
    w1b_reps = np.asarray(w1b_reps) if w1b_reps else np.zeros((0, len(conds)))
    lvl = float(ctx.cfg.stats.ci)
    wt = pd.DataFrame({"group": gnames, "weight": w1a,
                       "ci_lo": [percentile_ci(w1a_reps[:, i], lvl)[0] if len(w1a_reps) else np.nan for i in range(len(gnames))],
                       "ci_hi": [percentile_ci(w1a_reps[:, i], lvl)[1] if len(w1a_reps) else np.nan for i in range(len(gnames))],
                       "selection_pct": [100 * float(np.mean(w1a_reps[:, i] > 1e-6)) if len(w1a_reps) else np.nan for i in range(len(gnames))]})
    wc = pd.DataFrame({"condition": conds, "corruption": [lookup[c].corruption for c in conds],
                       "group": [lookup[c].group for c in conds], "severity": [lookup[c].severity for c in conds],
                       "weight": w1b,
                       "selection_pct": [100 * float(np.mean(w1b_reps[:, i] > 1e-6)) if len(w1b_reps) else np.nan for i in range(len(conds))]})

    # Level 2 + realism --------------------------------------------------------
    l2: dict = {}
    pi = np.full(len(conds), 1.0 / max(1, len(conds)))
    realism: dict = {}
    try:
        real, real_g, synth, sg, _ = load_phi(ctx, ds_name, conds)
        pi, info = level2_pi(ctx, real, synth)
        l2 = {k: v for k, v in info.items() if k != "proj"}
        uni_mix = {c: 1.0 / len(conds) for c in conds}
        wsum = w1a.sum()
        gw = {g: (w1a[i] / wsum if wsum > 0 else 1.0 / len(gnames)) for i, g in enumerate(gnames)}
        l1_mix = {}
        for g, wg_ in gw.items():
            members = [c for c in conds if lookup[c].group == g]
            for c in members:
                l1_mix[c] = wg_ / len(members)
        l2_mix = dict(zip(conds, pi))
        for name, mix in [("uniform", uni_mix), ("level1", l1_mix), ("level2", l2_mix)]:
            if sum(mix.values()) <= 0:
                continue
            realism[name] = realism_test(ctx, real, real_g, synth, sg, mix, 11, info["proj"], info["gamma"])
    except Exception as e:  # noqa: BLE001
        log.error("Level 2 / realism failed (features of the analysis encoder missing?): %s", e)
        l2 = {"error": str(e)}
    wc["pi"] = pi
    pred_l2 = D @ pi
    pred_l2_b = Db @ pi if Db is not None else None

    # Baselines ---------------------------------------------------------------
    b1 = np.nanmean(D, axis=1)
    b1b = np.nanmean(Db, axis=-1) if Db is not None else None
    if an.b2_conditions:
        B2 = delta_syn(an, conditions=an.b2_conditions)
        b2 = np.nanmean(B2, axis=1)
        b2b = np.nanmean(delta_syn(an, boot=True, conditions=an.b2_conditions), axis=-1) if Db is not None else None
    else:
        b2, b2b = np.full(len(y), np.nan), None
    s3 = [i for i, c in enumerate(conds) if lookup[c].severity == 3]
    G3 = np.stack([np.nanmean(D[:, [i for i in s3 if lookup[conds[i]].group == g]], axis=1) for g in gnames
                   if any(lookup[conds[i]].group == g for i in s3)], axis=1) if s3 else np.full((len(y), 1), np.nan)
    b5 = np.nanmean(G3, axis=1)
    if Db is not None and s3:
        G3b = np.stack([np.nanmean(Db[..., [i for i in s3 if lookup[conds[i]].group == g]], axis=-1) for g in gnames
                        if any(lookup[conds[i]].group == g for i in s3)], axis=-1)
        b5b = np.nanmean(G3b, axis=-1)
    else:
        b5b = None
    # B3: best single condition chosen on training backbones (LOBO), linear map to loss scale
    from realcal.stats.tests import spearman

    pred_b3 = np.full(len(y), np.nan)
    b3_choice: dict = {}
    for g in np.unique(bb[okc]):
        tr = okc & (bb != g)
        te = bb == g
        rhos = [spearman(D[tr, k], y[tr]) for k in range(D.shape[1])]
        rhos = np.nan_to_num(np.asarray(rhos), nan=-np.inf)
        k = int(np.argmax(rhos))
        if np.ptp(D[tr, k]) > 0:
            a, c = np.polyfit(D[tr, k], y[tr], 1)
        else:
            a, c = 0.0, float(np.mean(y[tr]))
        pred_b3[te] = a * D[te, k] + c
        b3_choice[str(g)] = (conds[k], a, c, k)
    pred_b3_b = None
    if Db is not None:
        pred_b3_b = np.full(Db.shape[:2], np.nan)
        for g, (_, a, c, k) in b3_choice.items():
            sel = bb == g
            pred_b3_b[:, sel] = a * Db[:, sel, k] + c
    # B4: clean HQ performance alone, LOBO linear
    clean = an.col("clean")
    pred_b4, coef_b4 = lobo_linear(clean, y, bb)
    pred_b4_b = None
    if an.boot is not None:
        cb = an.col("clean", boot=True)
        pred_b4_b = np.full(cb.shape, np.nan)
        for g, (a, c) in coef_b4.items():
            sel = bb == g
            pred_b4_b[:, sel] = a * cb[:, sel] + c

    preds = {"B1_uniform": b1, "B2_published": b2, "B3_best_single": pred_b3, "B4_clean_perf": pred_b4,
             "B5_balanced": b5, "RealCal_L1a_group": pred_1a, "RealCal_L1b_condition": pred_1b,
             "RealCal_L2_distribution": pred_l2, "RealCal_L1a_std": pred_1s}
    preds_b = {"B1_uniform": b1b, "B2_published": b2b, "B3_best_single": pred_b3_b, "B4_clean_perf": pred_b4_b,
               "B5_balanced": b5b,
               "RealCal_L1a_group": apply_fold_models(m1a, Gb, bb) if Gb is not None else None,
               "RealCal_L1b_condition": apply_fold_models(m1b, Db, bb) if Db is not None else None,
               "RealCal_L2_distribution": pred_l2_b,
               "RealCal_L1a_std": apply_std_models(m1s, Gb, bb) if Gb is not None else None}
    table = _methods_table(ctx, an, y, yb, preds, preds_b)
    table = mask_intercept_only(table, {"RealCal_L1a_group": m1a, "RealCal_L1b_condition": m1b,
                                        "RealCal_L1a_std": m1s})
    # MAE after LOBO linear calibration for raw-scale scores
    for name in ["B1_uniform", "B2_published", "B5_balanced", "RealCal_L2_distribution"]:
        cal, _ = lobo_linear(preds[name], y, bb)
        table.loc[table["method"] == name, "mae_lobo_calibrated"] = float(np.nanmean(np.abs(cal - y)))
    lofo = rank_stats(an, pred_lofo, y, apply_fold_models(m_lofo, Gb, fam) if Gb is not None else None, yb, lvl)

    write_csv(ctx.p("results", "exp3_validity_internal.csv"), table)
    write_csv(ctx.p("results", "exp3_group_weights.csv"), wt)
    write_csv(ctx.p("results", "exp3_condition_weights.csv"), wc)
    sc = config_frame(ctx, an)
    sc["delta_real_any"] = y
    for k, v in preds.items():
        sc[k] = v
    sc["RealCal_L1a_LOFO"] = pred_lofo
    write_csv(ctx.p("results", "exp3_scores.csv"), sc)
    out = {"lambda_1a": lam1a, "lambda_1b": lam1b, "intercept_1a": b1a, "group_weights": dict(zip(gnames, w1a.tolist())),
           "level2": l2, "realism": realism, "lofo": strip_boot(lofo),
           "lobo_lambdas_1a": {m.held_out: m.lam for m in m1a}, "b3_choices": {k: v[0] for k, v in b3_choice.items()},
           "intercept_only": {"level1a_full_fit": bool(np.all(w1a <= 1e-8)),
                              "level1b_full_fit": bool(np.all(w1b <= 1e-8)),
                              "level1a_std_full_fit": bool(np.all(w1s <= 1e-8)),
                              "level1a_lobo_folds": f"{n_intercept_only(m1a)}/{len(m1a)}",
                              "level1b_lobo_folds": f"{n_intercept_only(m1b)}/{len(m1b)}",
                              "level1a_std_lobo_folds": f"{n_intercept_only(m1s)}/{len(m1s)}"},
           "level1a_std": {"rel_lambda": lam1s, "intercept": b1s, "group_weights_z": dict(zip(gnames, w1s.tolist())),
                           "lobo_rel_lambdas": {m.held_out: m.lam for m in m1s}}}
    try:
        from realcal.experiments.checks import noise_ceiling

        nc = noise_ceiling(ctx, ds_name)
        out["noise_ceiling"] = nc
        write_json(ctx.p("results", "noise_ceiling.json"), nc)
    except Exception as e:  # noqa: BLE001
        log.warning("noise ceiling failed: %s", e)
    write_json(ctx.p("results", "exp3_summary.json"), out)

    frozen = {"intercept": b1a, "groups": gnames, "group_weights": w1a.tolist(), "lambda": lam1a,
              "level1b": {"intercept": int1b, "conditions": conds, "weights": w1b.tolist(), "lambda": lam1b},
              "level2": {"conditions": conds, "pi": pi.tolist(), "encoder": str(ctx.cfg.features.level2_encoder)},
              "intercept_only": bool(np.all(w1a <= 1e-8)),
              "level1a_std": {"intercept": b1s, "groups": gnames, "weights_z": w1s.tolist(), "mu": mu1s.tolist(),
                              "sd": sd1s.tolist(), "rel_lambda": lam1s},
              "condition_suite_hash": config_hash(ctx.cfg.corruption_suite),
              "config_hash": config_hash({k: ctx.cfg[k] for k in ("train", "threshold", "calibration", "corruption_suite")}),
              "code_hash": code_hash(), "metric": an.metric, "smoke": ctx.smoke}
    write_frozen(ctx, frozen)
    return out


def write_frozen(ctx: Ctx, frozen: dict) -> None:
    fp = ctx.p("prereg", "frozen_weights.json")
    lock = ctx.p("prereg", "stage2_lock.json")
    if lock.exists():
        old = read_json(fp) if fp.exists() else None
        if old is not None and config_hash(old) != config_hash(frozen):
            alt = ctx.p("prereg", "frozen_weights_rerun.json")
            write_json(alt, frozen)
            log.error("stage-2 lock exists: frozen_weights.json NOT overwritten; new fit saved to %s", alt)
            return
    write_json(fp, frozen)
    log.info("frozen weights -> %s", fp)


def frozen_scores(frozen: dict, ctx: Ctx, an: Analysis, boot: bool = False) -> dict[str, np.ndarray]:
    """Apply frozen RealCal weights (no refitting) to an analysis on any dataset."""
    D = delta_syn(an, boot=boot)
    G, gnames = group_scores(ctx, an, D)
    gi = [gnames.index(g) for g in frozen["groups"]]
    w = np.asarray(frozen["group_weights"])
    s1a = frozen["intercept"] + G[..., gi] @ w
    ci = [an.syn_conditions.index(c) for c in frozen["level1b"]["conditions"]]
    s1b = frozen["level1b"]["intercept"] + D[..., ci] @ np.asarray(frozen["level1b"]["weights"])
    ci2 = [an.syn_conditions.index(c) for c in frozen["level2"]["conditions"]]
    s2 = D[..., ci2] @ np.asarray(frozen["level2"]["pi"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        b1 = np.nanmean(D, axis=-1)
    out = {"B1_uniform": b1, "RealCal_L1a_group": s1a, "RealCal_L1b_condition": s1b, "RealCal_L2_distribution": s2}
    st = frozen.get("level1a_std")
    if st:
        gs = [gnames.index(g) for g in st["groups"]]
        out["RealCal_L1a_std"] = st["intercept"] + ((G[..., gs] - np.asarray(st["mu"])) / np.asarray(st["sd"])) @ np.asarray(st["weights_z"])
    if an.b2_conditions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            out["B2_published"] = np.nanmean(delta_syn(an, boot=boot, conditions=an.b2_conditions), axis=-1)
    return out


