"""Experiment 2: synthetic arm (75 conditions) and H1 (uniform benchmark vs real loss)."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.experiments.common import (REAL_STRATA, build_analysis, config_frame, delta_real, delta_syn,
                                        group_scores, rank_stats, strip_boot)
from realcal.stats.tests import zone
from realcal.utils.io import write_csv, write_json

log = logging.getLogger(__name__)


def nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def uniform_score(an, boot: bool = False) -> np.ndarray:
    return nanmean(delta_syn(an, boot=boot), axis=-1)


def run_exp2(ctx: Ctx, dataset: str = "deepdrid", train_ds: str | None = None, tag: str = "") -> dict:
    an = build_analysis(ctx, dataset, train_ds)
    if an is None:
        raise RuntimeError("no predictions; run predict-real and predict-synth first")
    lvl = float(ctx.cfg.stats.ci)
    ds = delta_syn(an)
    t = pd.DataFrame(ds, columns=an.syn_conditions)
    t.insert(0, "config", an.configs)
    write_csv(ctx.p("results", f"exp2_delta_syn{tag}.csv"), t)
    uni = uniform_score(an)
    uni_b = uniform_score(an, boot=True) if an.boot is not None else None
    gs, gnames = group_scores(ctx, an, ds)
    sc = config_frame(ctx, an)
    sc["uniform"] = uni
    for i, g in enumerate(gnames):
        sc[f"group_{g}"] = gs[:, i]
    if an.b2_conditions:
        sc["b2_mean"] = nanmean(delta_syn(an, conditions=an.b2_conditions), axis=-1)
    for q in REAL_STRATA:
        sc[f"delta_real_{q}"] = delta_real(an, q)
    sc["auroc_clean"] = an.col("clean")
    write_csv(ctx.p("results", f"exp2_scores{tag}.csv"), sc)
    res: dict = {"n_configs": len(an.configs), "n_conditions": len(an.syn_conditions)}
    zc = ctx.cfg.stats.zones
    for q in REAL_STRATA:
        r = rank_stats(an, uni, delta_real(an, q), uni_b,
                       delta_real(an, q, boot=True) if an.boot is not None else None, lvl)
        res[f"vs_real_{q}"] = strip_boot(r)
        if q == "any":
            lo, hi = r.get("spearman_ci", [np.nan, np.nan])
            res["zone"] = zone(lo, hi, float(zc.not_valid_upper), float(zc.valid_lower))
            res["rho"], res["rho_ci"] = r["spearman"], [lo, hi]
    write_json(ctx.p("results", f"h1{tag}.json"), res)
    log.info("H1%s: Spearman rho = %.3f CI %s -> %s", tag, res["rho"], res["rho_ci"], res["zone"])
    return res
