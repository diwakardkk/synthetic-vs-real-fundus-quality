"""Paper tables -> ``tables/<name>.csv`` and ``tables/<name>.tex`` (LaTeX tabular body, ready to paste)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.utils.io import atomic_write_text, read_csv_safe, read_json, write_csv

log = logging.getLogger(__name__)


def f(x, d: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "--"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def fci(x, lo, hi, d: int = 3) -> str:
    return f"{f(x, d)} ({f(lo, d)} to {f(hi, d)})"


def msd(m, s, d: int = 3) -> str:
    return f"{f(m, d)} $\\pm$ {f(s, d)}"


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def save_table(ctx: Ctx, name: str, df: pd.DataFrame, tex_df: pd.DataFrame | None = None) -> None:
    write_csv(ctx.p("tables", f"{name}.csv"), df)
    t = tex_df if tex_df is not None else df
    lines = [" & ".join(tex_escape(c) for c in t.columns) + " \\\\", "\\midrule"]
    for _, r in t.iterrows():
        lines.append(" & ".join(str(v) if "$" in str(v) else tex_escape(v) for v in r.values) + " \\\\")
    atomic_write_text(ctx.p("tables", f"{name}.tex"), "\n".join(lines) + "\n")


def _res(ctx: Ctx, name: str) -> Path:
    return ctx.out / "results" / name


def _csv(ctx: Ctx, name: str) -> pd.DataFrame | None:
    p = _res(ctx, name)
    return read_csv_safe(p)


def _json(ctx: Ctx, name: str) -> dict | None:
    p = _res(ctx, name)
    return read_json(p) if p.exists() else None


def table1(ctx: Ctx) -> None:
    rows = []
    for ds in ["deepdrid", "fives", "mshf", "ddr"]:
        p = ctx.out / "cache" / "metadata" / f"{ds}.parquet"
        if not p.exists():
            continue
        m = pd.read_parquet(p)
        rows.append({"Dataset": ds, "Images": len(m), "Patients": m["patient_id"].nunique(), "HQ": int(m["hq"].sum()),
                     "LQ": int(m["lq_any"].sum()), "Ungradable": int(m["ungradable"].sum()),
                     "Role": "development" if ds == "deepdrid" else "external"})
    if rows:
        save_table(ctx, "table1_datasets", pd.DataFrame(rows))


def table3(ctx: Ctx) -> None:
    p = ctx.out / "cache" / "metadata" / "deepdrid.parquet"
    if not p.exists():
        return
    m = pd.read_parquet(p)
    rows = []
    for name, mask in [("HQ", m["hq"]), ("LQ any", m["lq_any"]), ("LQ artifact", m["lq_artifact"]),
                       ("LQ clarity", m["lq_clarity"]), ("LQ field", m["lq_field"])]:
        s = m[mask]
        rows.append({"Stratum": name, "Images": len(s), "Patients": s["patient_id"].nunique(),
                     "Referable": int((s["label"] == 1).sum()),
                     "% referable": f(100 * (s["label"] == 1).mean() if len(s) else np.nan, 1)})
    v = _json(ctx, "viability.json") or {}
    lvl = v.get("pair_level", "patient")
    label = "Contralateral HQ/LQ eye pairs (same grade)" if lvl == "patient" else "Within-eye HQ/LQ pairs"
    rows.append({"Stratum": label, "Images": "--", "Patients": int(v.get("within_eye_pairs", 0)),
                 "Referable": "--", "% referable": "--"})
    save_table(ctx, "table3_strata", pd.DataFrame(rows))


def table4(ctx: Ctx) -> None:
    t = _csv(ctx, "exp1_real_loss.csv")
    if t is None:
        return
    s2 = _csv(ctx, "exp2_scores.csv")
    s3 = _csv(ctx, "exp3_scores.csv")
    if s2 is not None:
        t = t.merge(s2[["config", "uniform"]], on="config", how="left")
    if s3 is not None:
        t = t.merge(s3[["config", "RealCal_L1a_group"]], on="config", how="left")
    t = t.sort_values(["family", "backbone", "mode"])
    out = pd.DataFrame({
        "Backbone": t["backbone"], "Mode": t["mode"],
        "AUROC HQ": [msd(a, b) for a, b in zip(t["auroc_hq"], t["auroc_hq_sd"])],
        "AUROC LQ": [msd(a, b) for a, b in zip(t["auroc_lq_any"], t["auroc_lq_any_sd"])],
        "Real loss": [msd(a, b) for a, b in zip(t["delta_real_any"], t["delta_real_any_sd"])],
        "Uniform synthetic loss": [f(x) for x in t.get("uniform", pd.Series([np.nan] * len(t)))],
        "RealCal LOBO": [f(x) for x in t.get("RealCal_L1a_group", pd.Series([np.nan] * len(t)))],
        "Se HQ": [msd(a, b) for a, b in zip(t.get("se_hq", np.nan * np.ones(len(t))), t.get("se_hq_sd", np.nan * np.ones(len(t))))],
        "Se LQ": [msd(a, b) for a, b in zip(t.get("se_lq_any", np.nan * np.ones(len(t))), t.get("se_lq_sd", np.nan * np.ones(len(t))))],
    })
    save_table(ctx, "table4_main", t, out)


def table5(ctx: Ctx) -> None:
    w = _csv(ctx, "exp3_group_weights.csv")
    c = _csv(ctx, "exp3_condition_weights.csv")
    s = _json(ctx, "exp3_summary.json") or {}
    if w is None:
        return
    pi_g = c.groupby("group")["pi"].sum() if c is not None and "pi" in c else pd.Series(dtype=float)
    out = pd.DataFrame({"Group": w["group"], "Weight (95% CI)": [fci(a, b, cc, 4) for a, b, cc in zip(w["weight"], w["ci_lo"], w["ci_hi"])],
                        "Selected (%)": [f(x, 1) for x in w["selection_pct"]],
                        "Level-2 pi (group sum)": [f(pi_g.get(g, np.nan), 3) for g in w["group"]]})
    out.loc[len(out)] = ["Intercept", f(s.get("intercept_1a"), 4), "--", "--"]
    out.loc[len(out)] = ["Lambda", f(s.get("lambda_1a"), 4), "--", "--"]
    save_table(ctx, "table5_weights", out)


def table6(ctx: Ctx) -> None:
    t = _csv(ctx, "exp3_validity_internal.csv")
    if t is None:
        return
    fx = _csv(ctx, "exp4_fives_validity.csv")
    dr = _csv(ctx, "exp4_ddr_repl_validity.csv")
    rows = []
    for _, r in t.iterrows():
        row = {"Method": r["method"], "Internal rho (95% CI)": fci(r["spearman"], *_ci(r, "spearman")),
               "tau": f(r["kendall"]), "MAE": f(r["mae"], 4), "PRA": f(r["pra"]),
               "Intercept-only folds": (str(r.get("intercept_only_folds")) if isinstance(r.get("intercept_only_folds"), str)
                                        and r.get("intercept_only_folds") else "--")}
        if fx is not None and (fx.method == r["method"]).any():
            e = fx[fx.method == r["method"]].iloc[0]
            row.update({"FIVES rho (95% CI)": fci(e["spearman"], *_ci(e, "spearman")), "FIVES tau": f(e["kendall"]),
                        "FIVES PRA": f(e["pra"]), "FIVES Holm P": f(e.get("p_vs_B1_holm"), 3)})
        else:
            row.update({"FIVES rho (95% CI)": "--", "FIVES tau": "--", "FIVES PRA": "--", "FIVES Holm P": "--"})
        row["DDR repl. rho"] = f(dr[dr.method == r["method"]].iloc[0]["spearman"]) if dr is not None and (dr.method == r["method"]).any() else "--"
        rows.append(row)
    save_table(ctx, "table6_validity", pd.DataFrame(rows))
    s = _json(ctx, "exp3_summary.json") or {}
    lofo = s.get("lofo", {})
    if lofo:
        save_table(ctx, "supp_lofo", pd.DataFrame([{"Method": "RealCal L1a (LOFO)",
                                                    "rho (95% CI)": fci(lofo.get("spearman"), *(lofo.get("spearman_ci") or [np.nan, np.nan])),
                                                    "tau": f(lofo.get("kendall")), "MAE": f(lofo.get("mae"), 4), "PRA": f(lofo.get("pra"))}]))


def _ci(r, name):
    v = r.get(f"{name}_ci")
    if isinstance(v, str):
        try:
            v = [float(x) for x in v.strip("[]").split(",")]
        except ValueError:
            v = [np.nan, np.nan]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = [np.nan, np.nan]
    return v[0], v[1]


def table7(ctx: Ctx) -> None:
    rows = []
    s3 = _json(ctx, "exp3_summary.json") or {}
    for mix, r in (s3.get("realism") or {}).items():
        rows.append({"Test set": "DeepDRiD real LQ", "Analysis": "real vs synthetic AUROC", "Mix": mix,
                     "Value (95% CI)": fci(r["auroc"], *r["auroc_ci"]), "MMD2": f(r.get("mmd2"), 4)})
    m = _json(ctx, "exp4_mshf_summary.json") or {}
    for cam, d in (m.get("realism") or {}).items():
        for mix, r in d.items():
            rows.append({"Test set": f"MSHF {cam} LQ", "Analysis": "real vs synthetic AUROC", "Mix": mix,
                         "Value (95% CI)": fci(r["auroc"], *r["auroc_ci"]), "MMD2": f(r.get("mmd2"), 4)})
    for src, name in [(m, "MSHF"), (_json(ctx, "exp4_ddr_summary.json") or {}, "DDR")]:
        for k, q in (src.get("quality_detection") or {}).items():
            for mix in ["uniform", "realcal"]:
                rows.append({"Test set": f"{name} {k}", "Analysis": "quality detection AUROC", "Mix": mix,
                             "Value (95% CI)": fci(q[f"auroc_{mix}"], *q[f"auroc_{mix}_ci"]),
                             "MMD2": f"DeLong P={f(q.get('delong_p'), 3)}" if mix == "realcal" else "--"})
    if rows:
        save_table(ctx, "table7_realism_h3", pd.DataFrame(rows))


def table8(ctx: Ctx) -> None:
    s = _csv(ctx, "clinical_summary.csv")
    sel = _csv(ctx, "clinical_selection.csv")
    if s is None:
        return
    out = pd.DataFrame({"Stratum": s["stratum"], "Prevalence": [f"{100 * p:.0f}%" for p in s["prevalence"]],
                        "Missed per 1,000 (median, range)": [f"{f(a, 1)} ({f(b, 1)} to {f(c, 1)})" for a, b, c in
                                                             zip(s["missed_median"], s["missed_min"], s["missed_max"])],
                        "Unnecessary per 1,000 (median)": [f(x, 1) for x in s["unnecessary_median"]]})
    if sel is not None and len(sel):
        for _, r in sel.iterrows():
            out.loc[len(out)] = [f"LQ, selected by {r['selected_by']} ({r['config']})", f"{100 * r['prevalence']:.0f}%",
                                 f(r["missed_per_1000_LQ"], 1), f(r["unnecessary_per_1000_LQ"], 1)]
    save_table(ctx, "table8_clinical", out)


def supp_site(ctx: Ctx) -> None:
    t = _csv(ctx, "sens_subgroup_site.csv")
    if t is not None and len(t):
        g = t.groupby("site").agg(configs=("config", "nunique"), auroc_hq=("auroc_hq", "median"),
                                  auroc_lq=("auroc_lq", "median"), delta_real=("delta_real", "median")).reset_index()
        save_table(ctx, "supp_site_subgroups", g)


def make_tables(ctx: Ctx) -> None:
    for fn in (table1, table3, table4, table5, table6, table7, table8, supp_site):
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("table %s skipped: %s", fn.__name__, e)
    log.info("tables -> %s", ctx.out / "tables")
