"""Experiment 1: real arm on DeepDRiD (HQ vs real LQ), with confounding controls and calibration."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.eval.metrics import auroc, brier, ece_equal_mass, reliability
from realcal.eval.store import load_predictions
from realcal.experiments.common import REAL_STRATA, build_analysis, config_frame, delta_real
from realcal.stats.bootstrap import Engine, percentile_ci
from realcal.stats.mixed import gee_logit, mixed_logit
from realcal.stats.tests import mcnemar_exact
from realcal.utils.io import write_csv, write_json, write_parquet

log = logging.getLogger(__name__)


def real_loss_table(ctx: Ctx, dataset: str = "deepdrid", train_ds: str | None = None) -> pd.DataFrame:
    an = build_analysis(ctx, dataset, train_ds)
    ans = build_analysis(ctx, dataset, train_ds, metric="sens")
    if an is None:
        raise RuntimeError(f"no predictions for {dataset}; run predict-real first")
    t = config_frame(ctx, an)
    lvl = float(ctx.cfg.stats.ci)
    t["auroc_hq"], t["auroc_hq_sd"] = an.col("clean"), an.perf_sd[:, an.ci("clean")]
    for q in REAL_STRATA:
        t[f"auroc_lq_{q}"] = an.col(f"real_{q}")
        t[f"auroc_lq_{q}_sd"] = an.perf_sd[:, an.ci(f"real_{q}")]
        d = delta_real(an, q)
        db = delta_real(an, q, boot=True) if an.boot is not None else None
        t[f"delta_real_{q}"] = d
        t[f"delta_real_{q}_rel"] = delta_real(an, q, relative=True)
        sd = an.perf_seed[:, :, an.ci("clean")] - an.perf_seed[:, :, an.ci(f"real_{q}")]
        t[f"delta_real_{q}_sd"] = np.nanstd(sd, axis=1, ddof=1) if sd.shape[1] > 1 else 0.0
        if db is not None:
            cis = [percentile_ci(db[:, i], lvl) for i in range(len(an.configs))]
            t[f"delta_real_{q}_lo"] = [c[0] for c in cis]
            t[f"delta_real_{q}_hi"] = [c[1] for c in cis]
    if ans is not None:
        sidx = {c: i for i, c in enumerate(ans.configs)}
        ii = [sidx.get(c) for c in an.configs]
        get = lambda name: np.array([ans.col(name)[i] if i is not None else np.nan for i in ii])  # noqa: E731
        getsd = lambda name: np.array([ans.perf_sd[i, ans.ci(name)] if i is not None else np.nan for i in ii])  # noqa: E731
        t["se_hq"], t["se_hq_sd"] = get("clean"), getsd("clean")
        for q in REAL_STRATA:
            t[f"se_lq_{q}"] = get(f"real_{q}")
            t[f"delta_real_sens_{q}"] = t["se_hq"] - t[f"se_lq_{q}"]
        t["se_lq_sd"] = getsd("real_any")
    return t


def within_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy-at-threshold loss within grade strata, averaged with HQ grade-prevalence weights."""
    rows = []
    d = df[df["condition"].isin(["clean", "real"])].copy()
    d["correct"] = (d["pred_at_threshold"].astype(int) == d["label"].astype(int)).astype(float)
    d["lq"] = d["condition"] == "real"
    for (b, m, s), g in d.groupby(["backbone", "mode", "seed"]):
        hq = g[~g["lq"]]
        prev = hq["grade"].value_counts(normalize=True)
        losses, weights = [], []
        for gr, w in prev.items():
            a = g[(g["grade"] == gr) & ~g["lq"]]["correct"]
            c = g[(g["grade"] == gr) & g["lq"]]["correct"]
            if len(a) >= 5 and len(c) >= 5:
                losses.append(a.mean() - c.mean())
                weights.append(w)
        val = float(np.average(losses, weights=weights)) if weights else np.nan
        rows.append({"config": f"{b}/{m}", "seed": s, "within_grade_acc_loss": val, "n_grades_used": len(weights)})
    r = pd.DataFrame(rows)
    return r.groupby("config").agg(within_grade_acc_loss=("within_grade_acc_loss", "mean"),
                                   within_grade_acc_loss_sd=("within_grade_acc_loss", "std"),
                                   n_grades_used=("n_grades_used", "min")).reset_index()


def eye_pairs(df: pd.DataFrame, seed: int = 0, level: str = "patient") -> tuple[pd.DataFrame, dict]:
    """Paired HQ/LQ control: 2x2 correct/incorrect table and exact McNemar per configuration and pooled.

    level 'patient': contralateral eyes of the same patient (one all-HQ eye, one all-LQ eye, same grade),
    eye-level decision = referable if any image of the eye is positive at the frozen threshold.
    level 'eye': the two views of one eye (one HQ, one LQ), image-level decisions.
    """
    from realcal.data.splits import contralateral_pairs

    d = df[df["condition"].isin(["clean", "real"]) & (df["seed"] == seed)].copy()
    rows = []
    tot = np.zeros(4, int)
    for (b, m), g in d.groupby(["backbone", "mode"]):
        if level == "eye":
            g = g.assign(correct=(g["pred_at_threshold"].astype(int) == g["label"].astype(int)).astype(int))
            e = g.groupby("eye_id").agg(n=("image_id", "size"), n_hq=("hq", "sum"))
            eyes = e[(e["n"] == 2) & (e["n_hq"] == 1)].index
            gg = g[g["eye_id"].isin(eyes)]
            hq = gg[gg["hq"]].set_index("eye_id")["correct"]
            lq = gg[~gg["hq"]].set_index("eye_id")["correct"]
            j = pd.concat([hq.rename("hq"), lq.rename("lq")], axis=1).dropna()
        else:
            p = contralateral_pairs(g)
            j = pd.DataFrame({"hq": (p["pred_hq"].astype(int) == p["label_hq"].astype(int)).astype(int),
                              "lq": (p["pred_lq"].astype(int) == p["label_lq"].astype(int)).astype(int)})
        a = int(((j.hq == 1) & (j.lq == 1)).sum())
        bb = int(((j.hq == 1) & (j.lq == 0)).sum())
        c = int(((j.hq == 0) & (j.lq == 1)).sum())
        dd = int(((j.hq == 0) & (j.lq == 0)).sum())
        tot += [a, bb, c, dd]
        rows.append({"config": f"{b}/{m}", "n_pairs": len(j), "both_correct": a, "hq_only_correct": bb,
                     "lq_only_correct": c, "both_wrong": dd, "mcnemar_p": mcnemar_exact(bb, c)})
    pooled = {"pair_level": level, "n_pairs": int(tot.sum()), "both_correct": int(tot[0]), "hq_only_correct": int(tot[1]),
              "lq_only_correct": int(tot[2]), "both_wrong": int(tot[3]), "mcnemar_p": mcnemar_exact(int(tot[1]), int(tot[2])),
              "note": "pooled over configurations (descriptive; pairs are reused across configurations)"}
    return pd.DataFrame(rows), pooled


def label_noise_bound(ctx: Ctx, df: pd.DataFrame) -> pd.DataFrame:
    """Apparent AUROC loss on HQ images caused only by label flips at given rates."""
    rates = [float(r) for r in ctx.cfg.stats.label_noise_rates]
    reps = int(ctx.cfg.stats.label_noise_reps)
    rows = []
    hq = df[df["condition"] == "clean"]
    for (b, m, s), g in hq.groupby(["backbone", "mode", "seed"]):
        y = g["label"].astype(int).to_numpy()
        p = g["prob"].to_numpy()
        base = auroc(y, p)
        for r in rates:
            rng = np.random.default_rng(int(r * 1000) + s)
            losses = []
            for _ in range(reps):
                flip = rng.random(len(y)) < r
                yy = np.where(flip, 1 - y, y)
                losses.append(base - auroc(yy, p))
            rows.append({"config": f"{b}/{m}", "seed": s, "rate": r, "noise_auroc_loss": float(np.nanmean(losses))})
    t = pd.DataFrame(rows)
    return t.groupby(["config", "rate"])["noise_auroc_loss"].mean().reset_index()


def excluding_grades(ctx: Ctx, df: pd.DataFrame, excl=(1, 2)) -> pd.DataFrame:
    """delta_real(any) recomputed after removing borderline grades."""
    d = df[df["condition"].isin(["clean", "real"]) & ~df["grade"].isin(list(excl))]
    cells = {"clean": d["condition"] == "clean", "real_any": (d["condition"] == "real") & d["lq_any"].astype(bool)}
    eng = Engine(d, cells, metric="auroc")
    perf = eng.seed_mean(eng.compute())
    return pd.DataFrame({"config": eng.configs, "delta_real_any_excl_grades_1_2": perf[:, 0] - perf[:, 1]})


def calibration(ctx: Ctx, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nb = int(ctx.cfg.stats.ece_bins)
    rows, rel = [], []
    d = df[df["condition"].isin(["clean", "real"]) & df["label"].notna()]
    for (b, m, s), g in d.groupby(["backbone", "mode", "seed"]):
        for cond, gg in g.groupby("condition"):
            y = gg["label"].astype(int).to_numpy()
            p = gg["prob"].to_numpy()
            rows.append({"config": f"{b}/{m}", "seed": s, "stratum": "HQ" if cond == "clean" else "LQ",
                         "ece": ece_equal_mass(y, p, nb), "brier": brier(y, p)})
            if s == d["seed"].min():
                for pm, fy, n in reliability(y, p, nb):
                    rel.append({"config": f"{b}/{m}", "stratum": "HQ" if cond == "clean" else "LQ",
                                "mean_pred": pm, "frac_pos": fy, "n": n})
    t = pd.DataFrame(rows).groupby(["config", "stratum"]).agg(ece=("ece", "mean"), ece_sd=("ece", "std"),
                                                               brier=("brier", "mean"), brier_sd=("brier", "std")).reset_index()
    return t, pd.DataFrame(rel)


def run_exp1(ctx: Ctx) -> None:
    ds = "deepdrid"
    table = real_loss_table(ctx, ds)
    df = load_predictions(ctx, ds, conditions=["clean", "real"])
    df = df[df["label"].notna()]
    wg = within_grade(df)
    pairs, pooled = eye_pairs(df, seed=int(min(ctx.seeds())), level=str(ctx.ds_cfg(ds).get("pair_level", "patient")))
    noise = label_noise_bound(ctx, df)
    excl = excluding_grades(ctx, df)
    cal, rel = calibration(ctx, df)
    s0 = df[df["seed"] == min(ctx.seeds())].copy()
    s0["correct"] = (s0["pred_at_threshold"].astype(int) == s0["label"].astype(int)).astype(int)
    s0["lq"] = (s0["condition"] == "real").astype(int)
    s0["model"] = s0["backbone"] + "/" + s0["mode"]
    try:
        mixed = mixed_logit(s0)
    except Exception as e:  # noqa: BLE001
        log.warning("mixed model failed: %s", e)
        mixed = {"error": str(e)}
    try:
        gee = gee_logit(s0)
    except Exception as e:  # noqa: BLE001
        log.warning("GEE failed: %s", e)
        gee = {"error": str(e)}
    table = table.merge(wg, on="config", how="left").merge(excl, on="config", how="left")
    write_csv(ctx.p("results", "exp1_real_loss.csv"), table)
    write_csv(ctx.p("results", "exp1_within_eye_pairs.csv"), pairs)
    write_csv(ctx.p("results", "exp1_label_noise.csv"), noise)
    write_csv(ctx.p("results", "exp1_calibration.csv"), cal)
    write_parquet(ctx.p("results", "exp1_reliability.parquet"), rel)
    noise_med = noise.groupby("rate")["noise_auroc_loss"].median().to_dict()
    summary = {
        "n_configs": int(len(table)),
        "delta_real_any_median": float(np.nanmedian(table["delta_real_any"])),
        "delta_real_any_range": [float(np.nanmin(table["delta_real_any"])), float(np.nanmax(table["delta_real_any"]))],
        "delta_real_by_defect_median": {q: float(np.nanmedian(table[f"delta_real_{q}"])) for q in REAL_STRATA},
        "within_grade_acc_loss_median": float(np.nanmedian(table["within_grade_acc_loss"])),
        "within_eye_pairs_pooled": pooled,
        "mixed_model": mixed, "gee": gee,
        "label_noise_median_loss_by_rate": {str(k): float(v) for k, v in noise_med.items()},
        "delta_real_any_excl_grades_1_2_median": float(np.nanmedian(table["delta_real_any_excl_grades_1_2"])),
    }
    if "delta_real_sens_any" in table:
        summary["delta_real_sens_any_median"] = float(np.nanmedian(table["delta_real_sens_any"]))
    # case-mix check: LQ re-weighted to the HQ DR-grade distribution (quality is confounded with grade)
    try:
        from realcal.experiments.checks import standardized_real_loss, summarize_standardized

        std_tabs, std_sum = [], {}
        for q in REAL_STRATA:
            st = standardized_real_loss(ctx, ds, by="grade", stratum=q)
            if len(st):
                std_tabs.append(st)
                std_sum[q] = summarize_standardized(st)
        if std_tabs:
            write_csv(ctx.p("results", "exp1_grade_standardized.csv"), pd.concat(std_tabs, ignore_index=True))
        summary["grade_standardized"] = std_sum
    except Exception as e:  # noqa: BLE001
        log.warning("grade-standardised real loss failed: %s", e)
    write_json(ctx.p("results", "exp1_summary.json"), summary)
    log.info("Exp1: median delta_real(any) = %.4f", summary["delta_real_any_median"])
