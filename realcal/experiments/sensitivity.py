"""Sensitivity and subgroup analyses (analysis parts; GPU parts are launched from the CLI).

which:
* ``448``      - repeat at 448 px (variant 's448'): real loss and H1 at 448 vs 224.
* ``aug``      - corruption-augmented training (variant 'aug'): synthetic vs real loss change.
* ``sens``     - sensitivity-based (instead of AUROC-based) losses: H1 and RealCal internal validity.
* ``perfold``  - mean of per-fold metrics instead of pooled out-of-fold metrics.
* ``subgroup`` - DeepDRiD by site.
* ``seedsd``   - seed-to-seed standard deviations.
* ``noise``    - label-noise bound and borderline-grade exclusion (from Exp 1).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.corruptions.registry import active_main
from realcal.eval.store import load_predictions
from realcal.experiments.common import build_analysis, build_cells, delta_real, delta_syn, rank_stats, strip_boot
from realcal.stats.bootstrap import Engine
from realcal.stats.tests import spearman
from realcal.utils.io import read_json, write_csv, write_json

log = logging.getLogger(__name__)

ALL = ["448", "aug", "sens", "perfold", "subgroup", "seedsd", "noise"]


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def variant_vs_main(ctx: Ctx, variant: str) -> dict:
    an_v = build_analysis(ctx, "deepdrid", variant=variant)
    an_m = build_analysis(ctx, "deepdrid")
    if an_v is None or an_m is None:
        return {"error": f"no predictions for variant {variant}"}
    lvl = float(ctx.cfg.stats.ci)
    yv, uv = delta_real(an_v, "any"), _nanmean(delta_syn(an_v), -1)
    res = {"n_configs": len(an_v.configs),
           "h1": strip_boot(rank_stats(an_v, uv, yv, _nanmean(delta_syn(an_v, boot=True), -1) if an_v.boot is not None else None,
                                       delta_real(an_v, "any", boot=True) if an_v.boot is not None else None, lvl))}
    m_idx = [an_m.configs.index(c) for c in an_v.configs if c in an_m.configs]
    v_idx = [i for i, c in enumerate(an_v.configs) if c in an_m.configs]
    ym, um = delta_real(an_m, "any")[m_idx], _nanmean(delta_syn(an_m), -1)[m_idx]
    rows = pd.DataFrame({"config": [an_v.configs[i] for i in v_idx], "delta_real_main": ym, f"delta_real_{variant}": yv[v_idx],
                         "uniform_main": um, f"uniform_{variant}": uv[v_idx]})
    write_csv(ctx.p("results", f"sens_{variant}.csv"), rows)
    res["delta_real_change_median"] = float(np.nanmedian(yv[v_idx] - ym))
    res["uniform_change_median"] = float(np.nanmedian(uv[v_idx] - um))
    res["rank_agreement_real_main_vs_variant"] = spearman(ym, yv[v_idx])
    return res


def sens_based(ctx: Ctx) -> dict:
    from realcal.experiments.exp2_synth import run_exp2

    an = build_analysis(ctx, "deepdrid", metric="sens")
    if an is None:
        return {"error": "no predictions"}
    lvl = float(ctx.cfg.stats.ci)
    y = delta_real(an, "any")
    u = _nanmean(delta_syn(an), -1)
    r = rank_stats(an, u, y, _nanmean(delta_syn(an, boot=True), -1) if an.boot is not None else None,
                   delta_real(an, "any", boot=True) if an.boot is not None else None, lvl)
    _ = run_exp2
    return {"h1_sensitivity_based": strip_boot(r)}


def per_fold(ctx: Ctx) -> dict:
    syn = [c.name for c in active_main(ctx)]
    df = load_predictions(ctx, "deepdrid", conditions=["clean", "real"] + syn)
    df = df[df["label"].notna()]
    per = []
    for f, g in df.groupby("fold"):
        g = g.copy()
        cells = build_cells(g, syn)
        eng = Engine(g, cells, metric="auroc")
        perf = eng.seed_mean(eng.compute())
        dr = perf[:, eng.cells.index("clean")] - perf[:, eng.cells.index("real_any")]
        ds = perf[:, [eng.cells.index("clean")]] - perf[:, [eng.cells.index(s) for s in syn]]
        per.append(pd.DataFrame({"config": eng.configs, "fold": f, "delta_real": dr, "uniform": _nanmean(ds, 1)}))
    t = pd.concat(per).groupby("config")[["delta_real", "uniform"]].mean().reset_index()
    write_csv(ctx.p("results", "sens_perfold.csv"), t)
    return {"h1_rho_mean_of_folds": spearman(t["uniform"], t["delta_real"])}


def subgroup_site(ctx: Ctx) -> dict:
    df = load_predictions(ctx, "deepdrid", conditions=["clean", "real"])
    df = df[df["label"].notna() & df["site"].notna()]
    rows = []
    for site, g in df.groupby("site"):
        cells = build_cells(g, [])
        eng = Engine(g, {k: cells[k] for k in ("clean", "real_any")}, metric="auroc")
        perf = eng.seed_mean(eng.compute())
        for i, c in enumerate(eng.configs):
            rows.append({"site": site, "config": c, "auroc_hq": perf[i, 0], "auroc_lq": perf[i, 1],
                         "delta_real": perf[i, 0] - perf[i, 1], "n_hq": int((g.condition == "clean").sum() / max(1, g.seed.nunique() * g.groupby(['backbone','mode']).ngroups)),
                         "n_lq": int((g.condition == "real").sum() / max(1, g.seed.nunique() * g.groupby(['backbone','mode']).ngroups))})
    t = pd.DataFrame(rows)
    write_csv(ctx.p("results", "sens_subgroup_site.csv"), t)
    return {"sites": t.groupby("site")["delta_real"].median().to_dict()} if len(t) else {"sites": {}}


def seed_sd(ctx: Ctx) -> dict:
    an = build_analysis(ctx, "deepdrid")
    if an is None:
        return {}
    ps = an.perf_seed
    dr = ps[:, :, an.ci("clean")] - ps[:, :, an.ci("real_any")]
    ks = [an.ci(c) for c in an.syn_conditions]
    uni = _nanmean(ps[:, :, [an.ci("clean")]] - ps[:, :, ks], -1)
    t = pd.DataFrame({"config": an.configs, "auroc_hq_sd": np.nanstd(ps[:, :, an.ci("clean")], 1, ddof=1),
                      "delta_real_sd": np.nanstd(dr, 1, ddof=1), "uniform_sd": np.nanstd(uni, 1, ddof=1)})
    write_csv(ctx.p("results", "sens_seed_sd.csv"), t)
    return {"median_delta_real_sd": float(np.nanmedian(t["delta_real_sd"]))}


def run_sensitivity_analysis(ctx: Ctx, which: list[str]) -> dict:
    res: dict = {}
    for w in which:
        try:
            if w == "448":
                res["s448"] = variant_vs_main(ctx, "s448")
            elif w == "aug":
                res["aug"] = variant_vs_main(ctx, "aug")
            elif w == "sens":
                res["sens"] = sens_based(ctx)
            elif w == "perfold":
                res["perfold"] = per_fold(ctx)
            elif w == "subgroup":
                res["subgroup"] = subgroup_site(ctx)
            elif w == "seedsd":
                res["seedsd"] = seed_sd(ctx)
            elif w == "noise":
                p = ctx.out / "results" / "exp1_summary.json"
                res["noise"] = {k: v for k, v in read_json(p).items() if "noise" in k or "excl" in k} if p.exists() else {}
        except Exception as e:  # noqa: BLE001
            log.error("sensitivity '%s' failed: %s", w, e)
            res[w] = {"error": str(e)}
    p = ctx.p("results", "sensitivity_summary.json")
    old = read_json(p) if p.exists() else {}
    old.update(res)
    write_json(p, old)
    return res
