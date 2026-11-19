"""Power simulation for rank-correlation endpoints with n = 24 configurations in 8 backbone clusters.

The true config-level relation between a predictor score and the real loss is generated with a
Gaussian copula at a given Spearman rho, plus a backbone random effect; the observed real loss gets
measurement noise with the SD taken from the observed bootstrap variance of delta_real (default 0.01
if not yet available). Detection = lower limit of the cluster-bootstrap 95% CI > 0 (single
correlation) or of the CI of the difference > 0 (two dependent correlations).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from realcal.config import Ctx
from realcal.utils.io import read_csv_safe, write_csv, write_json

log = logging.getLogger(__name__)


def _rho_to_r(rho: float) -> float:
    return float(2 * np.sin(np.pi * rho / 6))  # Spearman -> Pearson for a Gaussian copula


def _sim_once(rng, n_bb, per_bb, rho1, rho2, noise_sd, icc, n_boot):
    n = n_bb * per_bb
    cl = np.repeat(np.arange(n_bb), per_bb)
    r1, r2 = _rho_to_r(rho1), _rho_to_r(rho2 if rho2 is not None else rho1)
    cov = np.array([[1, r1, r2], [r1, 1, r1 * r2], [r2, r1 * r2, 1]])
    cov = cov + 1e-6 * np.eye(3)
    z = rng.multivariate_normal(np.zeros(3), cov, size=n)
    u = rng.normal(0, np.sqrt(icc / max(1e-9, 1 - icc)), size=(n_bb, 3))[cl]
    z = z + u
    real = 0.05 + 0.03 * z[:, 0] + rng.normal(0, noise_sd, n)
    s1, s2 = z[:, 1], z[:, 2]
    rho_obs1 = stats.spearmanr(s1, real).statistic
    rho_obs2 = stats.spearmanr(s2, real).statistic
    b1, bd = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_bb, n_bb)
        idx = np.concatenate([np.flatnonzero(cl == k) for k in pick])
        a = stats.spearmanr(s1[idx], real[idx]).statistic
        b = stats.spearmanr(s2[idx], real[idx]).statistic
        b1.append(a)
        bd.append(b - a)
    lo1 = np.nanquantile(b1, 0.025)
    lod = np.nanquantile(bd, 0.025)
    return lo1 > 0, lod > 0, rho_obs1, rho_obs2


def run_power(ctx: Ctx) -> dict:
    pc = ctx.cfg.power
    n_sim = int(pc.n_sim)
    n_boot = 200 if not ctx.smoke else 20
    noise_sd = 0.01
    t4 = ctx.out / "results" / "exp1_real_loss.csv"
    t = read_csv_safe(t4)
    if t is not None:
        if {"delta_real_any_lo", "delta_real_any_hi"} <= set(t.columns):
            noise_sd = float(np.nanmedian((t["delta_real_any_hi"] - t["delta_real_any_lo"]) / (2 * 1.96)))
    n_bb, per_bb, icc = 8, 3, 0.3
    rows = []
    for rho in [float(r) for r in pc.rho_grid]:
        res = Parallel(n_jobs=int(ctx.cfg.stats.n_jobs))(
            delayed(_sim_once)(np.random.default_rng(1000 * int(rho * 100) + s), n_bb, per_bb, rho, None, noise_sd, icc, n_boot)
            for s in range(n_sim))
        rows.append({"analysis": "single", "rho": rho, "diff": 0.0, "power": float(np.mean([r[0] for r in res]))})
    base = 0.3
    for d in [float(x) for x in pc.diff_grid]:
        rho2 = min(0.99, base + d)
        res = Parallel(n_jobs=int(ctx.cfg.stats.n_jobs))(
            delayed(_sim_once)(np.random.default_rng(777 + 1000 * int(d * 100) + s), n_bb, per_bb, base, rho2, noise_sd, icc, n_boot)
            for s in range(n_sim))
        rows.append({"analysis": "difference", "rho": base, "diff": d, "power": float(np.mean([r[1] for r in res]))})
    t = pd.DataFrame(rows)
    write_csv(ctx.p("results", "power_simulation.csv"), t)
    single = t[t.analysis == "single"]
    diff = t[t.analysis == "difference"]
    mdr = single.loc[single.power >= 0.8, "rho"].min() if (single.power >= 0.8).any() else np.nan
    mdd = diff.loc[diff.power >= 0.8, "diff"].min() if (diff.power >= 0.8).any() else np.nan
    out = {"min_detectable_rho_80": mdr, "min_detectable_diff_80": mdd, "noise_sd": noise_sd,
           "n_configs": n_bb * per_bb, "n_clusters": n_bb, "icc": icc, "n_sim": n_sim, "baseline_rho_for_diff": base}
    write_json(ctx.p("results", "power_summary.json"), out)
    log.info("power: min detectable rho %.2f, min detectable difference %.2f", mdr, mdd)
    return out
