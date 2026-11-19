"""Experiment 4: external validation with frozen weights (nothing re-fitted).

* FIVES (primary endpoint): 24 configurations x 3 seeds retrained for 4-class classification on HQ
  official-train images; real loss (macro OvR AUROC) on official test; synthetic arm on HQ test
  images; frozen scores; Spearman(frozen RealCal, real loss) vs B1 with bootstrap difference.
* MSHF: realism per camera; H3 quality detectors.
* DDR: detectors on gradable vs ungradable; entropy / over-confidence of DeepDRiD models on
  ungradable images; replication (train on DDR, test on DeepDRiD).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.experiments.common import (build_analysis, build_cells, config_frame, delta_real, rank_stats,
                                        strip_boot)
from realcal.experiments.exp3_calibrate import frozen_scores
from realcal.prereg import require_lock
from realcal.stats.bootstrap import Engine, percentile_ci
from realcal.stats.tests import bootstrap_diff_p, holm, spearman
from realcal.utils.io import write_csv, write_json, write_parquet

log = logging.getLogger(__name__)


def evaluate_frozen(ctx: Ctx, frozen: dict, eval_ds: str, train_ds: str, variant: str = "main", tag: str = "") -> dict:
    an = build_analysis(ctx, eval_ds, train_ds, variant)
    if an is None:
        raise RuntimeError(f"no predictions for {eval_ds} (trained on {train_ds}, {variant})")
    lvl = float(ctx.cfg.stats.ci)
    y = delta_real(an, "any")
    yb = delta_real(an, "any", boot=True) if an.boot is not None else None
    sc = frozen_scores(frozen, ctx, an)
    scb = frozen_scores(frozen, ctx, an, boot=True) if an.boot is not None else {}
    rows, boots = [], {}
    for name, s in sc.items():
        r = rank_stats(an, s, y, scb.get(name), yb, lvl)
        boots[name] = r.get("spearman_boot")
        rows.append({"method": name, **strip_boot(r)})
    t = pd.DataFrame(rows)
    pv = []
    for i, r in t.iterrows():
        if r["method"] == "B1_uniform" or boots.get(r["method"]) is None or boots.get("B1_uniform") is None:
            pv.append(np.nan)
            continue
        d = np.asarray(boots[r["method"]]) - np.asarray(boots["B1_uniform"])
        lo, hi = percentile_ci(d, lvl)
        t.loc[i, "diff_vs_B1"] = r["spearman"] - t.loc[t.method == "B1_uniform", "spearman"].iloc[0]
        t.loc[i, "diff_vs_B1_lo"], t.loc[i, "diff_vs_B1_hi"] = lo, hi
        pv.append(bootstrap_diff_p(d))
    t["p_vs_B1"] = pv
    t["p_vs_B1_holm"] = holm(pv)
    write_csv(ctx.p("results", f"exp4_{tag}_validity.csv"), t)
    cf = config_frame(ctx, an)
    cf["delta_real_any"] = y
    cf["auroc_clean"] = an.col("clean")
    cf["auroc_lq"] = an.col("real_any")
    for k, v in sc.items():
        cf[k] = v
    write_csv(ctx.p("results", f"exp4_{tag}_scores.csv"), cf)
    prim = t[t.method == "RealCal_L1a_group"].iloc[0].to_dict() if (t.method == "RealCal_L1a_group").any() else {}
    return {"table": t.to_dict(orient="records"), "primary": prim}


def per_class_losses(ctx: Ctx, frozen: dict) -> pd.DataFrame:
    """FIVES per-class OvR real loss and its correlation with the frozen scores (point estimates)."""
    from realcal.eval.store import load_predictions
    from realcal.corruptions.registry import active_b2, active_main

    syn = [c.name for c in active_main(ctx)] + [c.name for c in active_b2(ctx)]
    df = load_predictions(ctx, "fives", conditions=["clean", "real"] + syn)
    df = df[df["label"].notna()]
    names = list(ctx.ds_cfg("fives").class_names)
    an = build_analysis(ctx, "fives")
    sc = frozen_scores(frozen, ctx, an)
    rows = []
    for k, cname in enumerate(names):
        d = df.copy()
        d["label"] = (d["label"].astype(int) == k).astype(int)
        d["prob"] = d[f"prob_{k}"]
        cells = build_cells(d, [])
        eng = Engine(d, {c: cells[c] for c in ("clean", "real_any")}, metric="auroc", n_classes=2)
        perf = eng.seed_mean(eng.compute())
        loss = perf[:, 0] - perf[:, 1]
        idx = [eng.configs.index(c) if c in eng.configs else None for c in an.configs]
        loss_al = np.array([loss[i] if i is not None else np.nan for i in idx])
        for m, s in sc.items():
            rows.append({"class": cname, "method": m, "spearman": spearman(s, loss_al)})
    return pd.DataFrame(rows)


def run_fives_analysis(ctx: Ctx) -> dict:
    frozen = require_lock(ctx)
    out = evaluate_frozen(ctx, frozen, "fives", "fives", "main", "fives")
    pc = per_class_losses(ctx, frozen)
    write_csv(ctx.p("results", "exp4_fives_per_class.csv"), pc)
    # case-mix check: FIVES LQ images are enriched for glaucoma -> re-weight LQ to the HQ class mix
    try:
        from realcal.experiments.checks import noise_ceiling, standardized_real_loss, summarize_standardized

        st = standardized_real_loss(ctx, "fives", by="label")
        if len(st):
            write_csv(ctx.p("results", "exp4_fives_class_standardized.csv"), st)
            out["class_standardized"] = summarize_standardized(st)
            sc = pd.read_csv(ctx.p("results", "exp4_fives_scores.csv"))
            m = st.merge(sc, on="config", how="left")
            out["class_standardized"]["spearman_vs_methods"] = {
                k: spearman(m[k].to_numpy(float), m["delta_std"].to_numpy(float))
                for k in sc.columns if k.startswith(("B1_", "B2_", "RealCal_"))}
        out["noise_ceiling"] = noise_ceiling(ctx, "fives")
    except Exception as e:  # noqa: BLE001
        log.warning("FIVES case-mix / noise-ceiling checks failed: %s", e)
    from realcal.data.common import load_metadata
    from realcal.data.splits import load_splits

    meta = load_metadata(ctx, "fives").merge(load_splits(ctx, "fives")[["image_id", "role_f0"]], on="image_id")
    n_lq_test = int((meta["lq_any"] & (meta["role_f0"] == "test")).sum())
    out["n_lq_test"] = n_lq_test
    cv_dir = ctx.out / "predictions" / "fives"
    if any(d.name.startswith("train__fives-cv__") for d in cv_dir.iterdir()) if cv_dir.exists() else False:
        out["cv_variant"] = evaluate_frozen(ctx, frozen, "fives", "fives", "cv", "fives_cv")
    write_json(ctx.p("results", "exp4_fives_summary.json"), out)
    return out


def run_mshf_analysis(ctx: Ctx) -> dict:
    """Realism of the three mixes against MSHF real LQ (per camera) and the H3 detectors."""
    from realcal.data.common import load_metadata
    from realcal.experiments.exp3_calibrate import _pca, _sqd, realism_test
    from realcal.experiments.quality_detector import evaluate
    from realcal.models.features import get_features

    frozen = require_lock(ctx)
    meta = load_metadata(ctx, "mshf")
    res: dict = {"realism": {}}
    enc = str(ctx.cfg.features.level2_encoder)
    from realcal.corruptions.cache import eval_hq_ids
    from realcal.corruptions.registry import by_name

    conds = list(frozen["level2"]["conditions"])
    lookup = by_name(ctx)
    hq = eval_hq_ids(ctx, "deepdrid")
    rng = np.random.default_rng(0)
    ids = sorted(rng.choice(hq, size=min(int(ctx.cfg.calibration.mmd_max_per_set), len(hq)), replace=False).tolist())
    dmeta = load_metadata(ctx, "deepdrid").set_index("image_id")
    synth = {c: get_features(ctx, enc, "deepdrid", 224, c, ids) for c in conds}
    sg = {c: dmeta.loc[ids, "patient_id"].astype(str).to_numpy() for c in conds}
    pooled = np.concatenate(list(synth.values()))
    proj = _pca(pooled[rng.choice(len(pooled), size=min(2000, len(pooled)), replace=False)])
    ps = proj(pooled[:500])
    d = _sqd(ps, ps)
    gamma = 1.0 / (np.median(d[d > 0]) if (d > 0).any() else 1.0)
    gw = dict(zip(frozen["groups"], frozen["group_weights"]))
    tot = sum(gw.values())
    l1 = {}
    for g, w in gw.items():
        mem = [c for c in conds if lookup[c].group == g]
        for c in mem:
            l1[c] = (w / tot if tot > 0 else 1 / len(gw)) / len(mem)
    mixes = {"uniform": {c: 1 / len(conds) for c in conds}, "level1": l1, "level2": dict(zip(conds, frozen["level2"]["pi"]))}
    for cam, g in meta[meta["lq_any"]].groupby("camera"):
        real = get_features(ctx, enc, "mshf", 224, "clean", g["image_id"].tolist())
        rg = ("mshf_" + g["patient_id"].astype(str)).to_numpy()
        res["realism"][str(cam)] = {k: realism_test(ctx, real, rg, synth, sg, mx, 23, proj, gamma)
                                    for k, mx in mixes.items() if sum(mx.values()) > 0}
    q, scores = evaluate(ctx, frozen, "mshf", meta, "lq_any", by="camera")
    res["quality_detection"] = q
    # sensitivity: drop source folders whose quality is near-constant (Glaucoma ~98% LQ, Healthy 100% HQ),
    # so that detection cannot ride on disease/source differences
    excl = [str(x) for x in ctx.cfg.get("external", {}).get("mshf_confounded_sites", ["Glaucoma", "Healthy"])]
    try:
        mb = meta[~meta["site"].astype(str).isin(excl)].copy()
        qb, _ = evaluate(ctx, frozen, "mshf", mb, "lq_any", by="camera")
        res["quality_detection_balanced_sites"] = {"excluded_sites": excl, "n_images": int(len(mb)),
                                                   "n_lq": int(mb["lq_any"].sum()), **qb}
    except Exception as e:  # noqa: BLE001
        log.warning("MSHF balanced-site detector analysis failed: %s", e)
    write_parquet(ctx.p("results", "exp4_mshf_detector_scores.parquet"), scores)
    write_json(ctx.p("results", "exp4_mshf_summary.json"), res)
    return res


def run_ddr_analysis(ctx: Ctx) -> dict:
    """Detectors on DDR gradable vs ungradable; entropy/over-confidence of DeepDRiD models."""
    from realcal.data.common import load_metadata
    from realcal.eval.metrics import entropy
    from realcal.eval.store import load_predictions
    from realcal.experiments.quality_detector import evaluate
    from scipy.stats import mannwhitneyu

    frozen = require_lock(ctx)
    meta = load_metadata(ctx, "ddr")
    meta["ungradable_f"] = meta["ungradable"].astype(float)
    q, scores = evaluate(ctx, frozen, "ddr", meta, "ungradable_f")
    res: dict = {"quality_detection": q}
    write_parquet(ctx.p("results", "exp4_ddr_detector_scores.parquet"), scores)
    df = load_predictions(ctx, "ddr", train_dataset="deepdrid")
    if not df.empty:
        pcols = [c for c in df.columns if c.startswith("prob_")]
        df["entropy"] = entropy(df[pcols].to_numpy())
        df["maxp"] = df[pcols].max(axis=1)
        thr = float(ctx.cfg.external.confident_threshold)
        rows = []
        for (b, m), g in df.groupby(["backbone", "mode"]):
            u, gr = g[g["ungradable"].astype(bool)], g[~g["ungradable"].astype(bool)]
            p = mannwhitneyu(u["entropy"], gr["entropy"]).pvalue if len(u) and len(gr) else np.nan
            rows.append({"config": f"{b}/{m}", "entropy_ungradable": u["entropy"].mean(), "entropy_gradable": gr["entropy"].mean(),
                         "share_ungradable_confident": float((u["maxp"] > thr).mean()) if len(u) else np.nan,
                         "share_gradable_confident": float((gr["maxp"] > thr).mean()) if len(gr) else np.nan,
                         "mannwhitney_p": p})
        t = pd.DataFrame(rows)
        write_csv(ctx.p("results", "exp4_ddr_entropy.csv"), t)
        write_parquet(ctx.p("results", "exp4_ddr_entropy_rows.parquet"),
                      df[["backbone", "mode", "image_id", "ungradable", "entropy", "maxp"]])
        res["entropy"] = {"median_share_ungradable_confident": float(t["share_ungradable_confident"].median()),
                          "median_entropy_ungradable": float(t["entropy_ungradable"].median()),
                          "median_entropy_gradable": float(t["entropy_gradable"].median())}
    write_json(ctx.p("results", "exp4_ddr_summary.json"), res)
    return res


def run_ddr_repl_analysis(ctx: Ctx) -> dict:
    """DDR-trained models tested on DeepDRiD: ranking agreement with Exp 1 and frozen scores."""
    frozen = require_lock(ctx)
    out = evaluate_frozen(ctx, frozen, "deepdrid", "ddr", "main", "ddr_repl")
    an_r = build_analysis(ctx, "deepdrid", "ddr")
    an_d = build_analysis(ctx, "deepdrid")
    y_r = pd.Series(delta_real(an_r, "any"), index=an_r.configs)
    y_d = pd.Series(delta_real(an_d, "any"), index=an_d.configs)
    common = [c for c in an_d.configs if c in y_r.index]
    out["ranking_vs_exp1_spearman"] = spearman(y_r[common].to_numpy(), y_d[common].to_numpy())
    write_json(ctx.p("results", "exp4_ddr_repl_summary.json"), out)
    return out


__all__ = ["run_fives_analysis", "run_mshf_analysis", "run_ddr_analysis", "run_ddr_repl_analysis", "rank_stats"]
