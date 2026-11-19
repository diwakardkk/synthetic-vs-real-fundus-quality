"""Clinical translation: missed / unnecessary referrals per 1,000 screened, model selection, decision curves."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.eval.store import load_predictions
from realcal.experiments.common import build_analysis
from realcal.utils.io import read_csv_safe, read_json, write_csv, write_json

log = logging.getLogger(__name__)


def per_1000(se: float, sp: float, p: float) -> tuple[float, float]:
    return 1000 * p * (1 - se), 1000 * (1 - p) * (1 - sp)


def net_benefit(y: np.ndarray, risk: np.ndarray, pts: np.ndarray, prevalence: float | None = None) -> np.ndarray:
    y = np.asarray(y).astype(int)
    p = y.mean() if prevalence is None else prevalence
    out = []
    for pt in pts:
        pred = risk >= pt
        se = pred[y == 1].mean() if (y == 1).any() else np.nan
        sp = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
        out.append(se * p - (1 - sp) * (1 - p) * pt / (1 - pt))
    return np.asarray(out)


def select_models(ctx: Ctx, dataset: str, sc: pd.DataFrame | None) -> dict[str, tuple[str | None, str]]:
    """Model selection for deployment on low-quality images.

    Every rule picks the configuration with the highest *predicted LQ AUROC* = clean (HQ) AUROC minus
    the predicted loss of that rule (LOBO predictions from exp3, so nothing is fitted on the held-out
    backbone). ``clean_only`` ignores robustness; ``oracle`` uses the observed LQ AUROC (upper bound).
    Picking by smallest loss alone would reward weak-but-stable models, so it is not used.
    """
    out: dict[str, tuple[str | None, str]] = {}
    an = build_analysis(ctx, dataset)  # AUROC analysis (cached)
    if an is None:
        return out
    perf = pd.DataFrame({"config": an.configs, "auroc_hq": an.col("clean"), "auroc_lq": an.col("real_any")})
    out["clean_only"] = (perf.loc[perf["auroc_hq"].idxmax(), "config"], "max HQ AUROC")
    out["oracle"] = (perf.loc[perf["auroc_lq"].idxmax(), "config"], "max observed LQ AUROC (upper bound)")
    if sc is None:
        log.warning("clinical: exp3_scores.csv missing/empty -> only clean_only and oracle selections")
        return out
    m = perf.merge(sc, on="config", how="left")
    e3 = read_json(ctx.out / "results" / "exp3_summary.json") if (ctx.out / "results" / "exp3_summary.json").exists() else {}
    io = (e3.get("intercept_only") or {})
    for how, col in [("uniform", "B1_uniform"), ("realcal", "RealCal_L1a_group"), ("realcal_std", "RealCal_L1a_std")]:
        if col not in m or m[col].isna().all():
            continue
        note = "max (HQ AUROC - predicted loss)"
        key = {"realcal": "level1a_lobo_folds", "realcal_std": "level1a_std_lobo_folds"}.get(how)
        if key and str(io.get(key, "")).split("/")[0] == str(io.get(key, "")).split("/")[-1] and io.get(key):
            note += "; intercept-only model -> equivalent to clean_only up to a constant per fold"
        pred_lq = m["auroc_hq"] - m[col]
        out[how] = (m.loc[pred_lq.idxmax(), "config"], note)
    return out


def run_clinical(ctx: Ctx, dataset: str = "deepdrid") -> dict:
    an = build_analysis(ctx, dataset, metric="sens")
    if an is None:
        raise RuntimeError("no predictions")
    eng = an.engine
    se = eng.seed_mean(eng.compute(metric="sens"))
    sp = eng.seed_mean(eng.compute(metric="spec"))
    prevs = [float(p) for p in ctx.cfg.clinical.prevalences]
    rows = []
    for i, c in enumerate(an.configs):
        for stratum, cell in [("HQ", "clean"), ("LQ", "real_any")]:
            k = an.ci(cell)
            for p in prevs:
                m, u = per_1000(se[i, k], sp[i, k], p)
                rows.append({"config": c, "stratum": stratum, "prevalence": p, "sensitivity": se[i, k],
                             "specificity": sp[i, k], "missed_per_1000": m, "unnecessary_per_1000": u})
    t = pd.DataFrame(rows)
    write_csv(ctx.p("results", "clinical_missed.csv"), t)
    summ = t.groupby(["stratum", "prevalence"]).agg(missed_median=("missed_per_1000", "median"),
                                                    missed_min=("missed_per_1000", "min"),
                                                    missed_max=("missed_per_1000", "max"),
                                                    unnecessary_median=("unnecessary_per_1000", "median")).reset_index()
    write_csv(ctx.p("results", "clinical_summary.csv"), summ)

    sel_rows = []
    sp_path = ctx.out / "results" / "exp3_scores.csv"
    sc = read_csv_safe(sp_path)
    picks = select_models(ctx, dataset, sc)
    for how, (cfg, note) in picks.items():
        if cfg is None:
            continue
        for p in prevs:
            r = t[(t.config == cfg) & (t.stratum == "LQ") & (t.prevalence == p)].iloc[0]
            sel_rows.append({"selected_by": how, "config": cfg, "prevalence": p, "note": note,
                             "missed_per_1000_LQ": r.missed_per_1000, "unnecessary_per_1000_LQ": r.unnecessary_per_1000})
    sel = pd.DataFrame(sel_rows, columns=["selected_by", "config", "prevalence", "note", "missed_per_1000_LQ",
                                          "unnecessary_per_1000_LQ"])
    if not len(sel):
        log.warning("clinical: exp3_scores.csv missing/empty -> no model-selection rows")
    write_csv(ctx.p("results", "clinical_selection.csv"), sel)

    # decision curves
    cc = ctx.cfg.clinical
    pts = np.linspace(float(cc.dca_pt_min), float(cc.dca_pt_max), int(cc.dca_n))
    df = load_predictions(ctx, dataset, conditions=["clean", "real"])
    df = df[df["label"].notna()]
    configs = sorted(set(sel["config"])) if len(sel) else an.configs[:3]
    dca = []
    for cfg in configs:
        b, m = cfg.split("/")
        g = df[(df.backbone == b) & (df["mode"] == m)]
        for cond, stratum in [("clean", "HQ"), ("real", "LQ")]:
            for p in prevs:
                nbs = [net_benefit(gg["label"].astype(int).to_numpy(), gg["prob"].to_numpy(), pts, p)
                       for _, gg in g[g.condition == cond].groupby("seed")]
                if not nbs:
                    continue
                nb = np.nanmean(np.stack(nbs), axis=0)
                for pt, v in zip(pts, nb):
                    dca.append({"config": cfg, "stratum": stratum, "prevalence": p, "pt": pt, "net_benefit": v})
    for p in prevs:
        for pt in pts:
            dca.append({"config": "refer_all", "stratum": "all", "prevalence": p, "pt": pt, "net_benefit": p - (1 - p) * pt / (1 - pt)})
            dca.append({"config": "refer_none", "stratum": "all", "prevalence": p, "pt": pt, "net_benefit": 0.0})
    write_csv(ctx.p("results", "clinical_dca.csv"), pd.DataFrame(dca))
    out = {"summary": summ.to_dict(orient="records"), "selection": sel.to_dict(orient="records")}
    write_json(ctx.p("results", "clinical_summary.json"), out)
    _ = read_json
    return out
