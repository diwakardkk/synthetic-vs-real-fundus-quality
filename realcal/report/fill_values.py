"""``results/values.tex`` (one \\newcommand per number quoted in the text) and ``results/summary.md``."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.utils.io import atomic_write_text, read_csv_safe, read_json
from realcal.utils.provenance import gpu_hours

log = logging.getLogger(__name__)


def _j(ctx: Ctx, name: str) -> dict:
    p = ctx.out / "results" / name
    return read_json(p) if p.exists() else {}


def _c(ctx: Ctx, name: str) -> pd.DataFrame | None:
    p = ctx.out / "results" / name
    return read_csv_safe(p)


def _num(x, d=2) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def _parse_ci(v):
    if isinstance(v, str):
        try:
            return [float(z) for z in v.strip("[]").split(",")]
        except ValueError:
            return [np.nan, np.nan]
    return v if v is not None else [np.nan, np.nan]


class Values:
    def __init__(self):
        self.macros: list[tuple[str, str, str]] = []

    def add(self, name: str, value, desc: str, d: int = 2) -> None:
        assert re.fullmatch(r"[A-Za-z]+", name), name
        self.macros.append((name, value if isinstance(value, str) else _num(value, d), desc))

    def ci(self, name: str, est, ci, desc: str, d: int = 2) -> None:
        ci = _parse_ci(ci)
        self.add(name, est, desc, d)
        self.add(name + "Lo", ci[0], desc + " (lower 95% CI)", d)
        self.add(name + "Hi", ci[1], desc + " (upper 95% CI)", d)


def collect(ctx: Ctx) -> Values:
    v = Values()
    via = _j(ctx, "viability.json")
    if via:
        lvl = via.get("pair_level", "patient")
        v.add("NEyePairs", str(via.get("within_eye_pairs")),
              "contralateral HQ/LQ eye pairs (same patient, same grade)" if lvl == "patient" else "within-eye HQ/LQ pairs")
        v.add("NLQReferable", str(via.get("lq_referable")), "LQ images with referable DR")
    e1 = _j(ctx, "exp1_summary.json")
    if e1:
        v.add("DeltaRealMedian", e1.get("delta_real_any_median"), "median real AUROC loss (any defect)", 3)
        lo, hi = e1.get("delta_real_any_range", [np.nan, np.nan])
        v.add("DeltaRealMin", lo, "min real AUROC loss", 3)
        v.add("DeltaRealMax", hi, "max real AUROC loss", 3)
        for q, val in (e1.get("delta_real_by_defect_median") or {}).items():
            v.add("DeltaReal" + q.capitalize(), val, f"median real AUROC loss ({q})", 3)
        mm = e1.get("mixed_model", {})
        if "or" in mm:
            v.ci("MixedOR", mm["or"], mm.get("ci"), "odds ratio of correct classification for LQ (mixed model)")
        g = e1.get("gee", {})
        if "or" in g:
            v.ci("GeeOR", g["or"], g.get("ci"), "odds ratio for LQ (GEE)")
        pp = e1.get("within_eye_pairs_pooled", {})
        if pp:
            v.add("PairsHQOnly", str(pp.get("hq_only_correct")), "pairs correct only on HQ")
            v.add("PairsLQOnly", str(pp.get("lq_only_correct")), "pairs correct only on LQ")
            v.add("PairsMcNemarP", pp.get("mcnemar_p"), "pooled McNemar P", 3)
        for q, gs in (e1.get("grade_standardized") or {}).items():
            if gs:
                v.add("DeltaRealGradeStd" + q.capitalize(), gs.get("delta_std_median"),
                      f"median real AUROC loss ({q}), LQ re-weighted to the HQ grade mix", 3)
        if "delta_real_sens_any_median" in e1:
            v.add("DeltaRealSensMedian", e1["delta_real_sens_any_median"], "median sensitivity loss on LQ", 3)
    h1 = _j(ctx, "h1.json")
    if h1:
        v.ci("HoneRho", h1.get("rho"), h1.get("rho_ci"), "H1 Spearman rho uniform vs real")
        v.add("HoneZone", str(h1.get("zone")), "H1 pre-registered zone")
        a = h1.get("vs_real_any", {})
        v.ci("HoneTau", a.get("kendall"), a.get("kendall_ci"), "H1 Kendall tau")
        v.add("HonePRA", a.get("pra"), "H1 pairwise ranking accuracy")
        v.add("HoneTopFive", a.get("top5"), "H1 top-5 overlap")
    t = _c(ctx, "exp3_validity_internal.csv")
    if t is not None:
        names = {"B1_uniform": "BOne", "RealCal_L1a_group": "RealCalGroup", "RealCal_L1b_condition": "RealCalCond",
                 "RealCal_L2_distribution": "RealCalDist", "B2_published": "BTwo", "B3_best_single": "BThree",
                 "B4_clean_perf": "BFour", "B5_balanced": "BFive", "RealCal_L1a_std": "RealCalStd"}
        for _, r in t.iterrows():
            nm = names.get(r["method"])
            if not nm:
                continue
            v.ci(f"Int{nm}Rho", r["spearman"], r.get("spearman_ci"), f"internal LOBO Spearman rho, {r['method']}")
            v.add(f"Int{nm}MAE", r["mae"], f"internal MAE, {r['method']}", 4)
            v.add(f"Int{nm}PRA", r["pra"], f"internal PRA, {r['method']}")
            if r["method"] != "B1_uniform":
                v.add(f"Int{nm}DiffP", r.get("p_vs_B1_holm"), f"Holm P vs B1, {r['method']}", 3)
    e3 = _j(ctx, "exp3_summary.json")
    if e3:
        for gname, w in (e3.get("group_weights") or {}).items():
            v.add("W" + gname.capitalize(), w, f"Level-1a weight, {gname}", 4)
        v.add("LambdaOneA", e3.get("lambda_1a"), "selected lambda (Level 1a)", 4)
        io = e3.get("intercept_only") or {}
        if io:
            v.add("RealCalInterceptOnly", "yes" if io.get("level1a_full_fit") else "no",
                  "Level-1a full-data fit is intercept-only (all weights zero)")
            v.add("RealCalInterceptOnlyFolds", str(io.get("level1a_lobo_folds")),
                  "LOBO folds in which Level 1a is intercept-only")
            v.add("RealCalStdInterceptOnlyFolds", str(io.get("level1a_std_lobo_folds")),
                  "LOBO folds in which the standardised Level 1a is intercept-only")
        nc = e3.get("noise_ceiling") or {}
        if nc and "error" not in nc:
            v.add("RelPatient", nc.get("reliability_patient"), "split-half (patient) reliability of the real-loss ranking")
            v.add("RelSeed", nc.get("reliability_seed"), "seed reliability of the real-loss ranking")
            v.add("RelCombined", nc.get("reliability_combined"), "combined reliability of the real-loss ranking")
            v.add("CeilingRho", nc.get("ceiling_rho_combined"), "noise ceiling: maximum attainable Spearman rho")
            v.add("LossSDConfigs", nc.get("loss_sd_between_configs"), "SD of real loss between configurations", 4)
            v.add("LossSDSeed", nc.get("median_seed_sd"), "median seed SD of real loss", 4)
        for mix, r in (e3.get("realism") or {}).items():
            v.ci("Realism" + mix.capitalize().replace("1", "One").replace("2", "Two"), r.get("auroc"), r.get("auroc_ci"),
                 f"real-vs-synthetic AUROC, {mix} mix")
    fx = _c(ctx, "exp4_fives_validity.csv")
    if fx is not None:
        for _, r in fx.iterrows():
            nm = {"B1_uniform": "BOne", "RealCal_L1a_group": "RealCalGroup", "RealCal_L1b_condition": "RealCalCond",
                  "RealCal_L2_distribution": "RealCalDist", "B2_published": "BTwo"}.get(r["method"])
            if nm:
                v.ci(f"Fives{nm}Rho", r["spearman"], r.get("spearman_ci"), f"FIVES Spearman rho, {r['method']}")
            if r["method"] == "RealCal_L1a_group":
                v.ci("FivesPrimaryDiff", r.get("diff_vs_B1"), [r.get("diff_vs_B1_lo"), r.get("diff_vs_B1_hi")],
                     "FIVES primary endpoint: rho(RealCal) - rho(B1)")
                v.add("FivesPrimaryP", r.get("p_vs_B1"), "FIVES primary endpoint P", 3)
    for src, pre in [("exp4_mshf_summary.json", "Mshf"), ("exp4_ddr_summary.json", "Ddr")]:
        s = _j(ctx, src)
        q = (s.get("quality_detection") or {}).get("overall")
        if q:
            v.ci(pre + "DetUniform", q["auroc_uniform"], q["auroc_uniform_ci"], f"{pre} detector AUROC, uniform mix")
            v.ci(pre + "DetRealCal", q["auroc_realcal"], q["auroc_realcal_ci"], f"{pre} detector AUROC, RealCal mix")
            v.add(pre + "DetP", q["delong_p"], f"{pre} DeLong P", 3)
        if pre == "Ddr" and s.get("entropy"):
            v.add("DdrConfidentShare", 100 * s["entropy"]["median_share_ungradable_confident"],
                  "median % of ungradable DDR images with max probability > 0.9", 1)
    fs = _j(ctx, "exp4_fives_summary.json")
    if fs.get("class_standardized"):
        v.add("FivesDeltaClassStd", fs["class_standardized"].get("delta_std_median"),
              "FIVES median real loss, LQ re-weighted to the HQ class mix", 3)
        v.add("FivesDeltaRaw", fs["class_standardized"].get("delta_raw_median"), "FIVES median real loss (raw)", 3)
    if (fs.get("noise_ceiling") or {}).get("ceiling_rho_combined") is not None:
        v.add("FivesCeilingRho", fs["noise_ceiling"]["ceiling_rho_combined"], "FIVES noise ceiling (max attainable rho)")
    ms = _j(ctx, "exp4_mshf_summary.json")
    qb = (ms.get("quality_detection_balanced_sites") or {}).get("overall")
    if qb:
        v.ci("MshfBalDetUniform", qb["auroc_uniform"], qb["auroc_uniform_ci"], "MSHF detector AUROC (balanced sites), uniform mix")
        v.ci("MshfBalDetRealCal", qb["auroc_realcal"], qb["auroc_realcal_ci"], "MSHF detector AUROC (balanced sites), RealCal mix")
    rp = _j(ctx, "exp4_ddr_repl_summary.json")
    if rp:
        v.add("DdrReplRankRho", rp.get("ranking_vs_exp1_spearman"), "DDR replication: rank agreement with Exp 1")
    cl = _c(ctx, "clinical_summary.csv")
    if cl is not None:
        for _, r in cl.iterrows():
            pn = {0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}.get(round(float(r["prevalence"]), 2), "X")
            v.add(f"Missed{r['stratum']}{pn}", r["missed_median"], f"median missed per 1,000, {r['stratum']}, prevalence {r['prevalence']}", 1)
    sel = _c(ctx, "clinical_selection.csv")
    if sel is not None:
        for _, r in sel[sel.prevalence.round(2) == 0.1].iterrows():
            key = "".join(x.capitalize() for x in str(r["selected_by"]).split("_"))
            v.add("SelMissed" + key, r["missed_per_1000_LQ"],
                  f"missed per 1,000 on LQ at 10% prevalence, model selected by {r['selected_by']}", 1)
    pw = _j(ctx, "power_summary.json")
    if pw:
        v.add("PowerMinRho", pw.get("min_detectable_rho_80"), "minimum detectable rho (80% power)")
        v.add("PowerMinDiff", pw.get("min_detectable_diff_80"), "minimum detectable difference in rho (80% power)")
    gh = gpu_hours(ctx.out)
    v.add("GpuHours", gh["total_gpu_hours"], "total GPU hours", 0)
    return v


def write_values(ctx: Ctx) -> None:
    v = collect(ctx)
    tex = ["% Auto-generated by realcal.report.fill_values. Do not edit by hand."]
    md = ["# RealCal: headline numbers", "", "Plain-language list of every number quoted in the paper (95% CI where given).", ""]
    seen = set()
    for name, val, desc in v.macros:
        if name in seen:
            continue
        seen.add(name)
        tex.append(f"\\newcommand{{\\{name}}}{{{val}}}  % {desc}")
        md.append(f"- **{desc}**: {val}  (`\\{name}`)")
    atomic_write_text(ctx.p("results", "values.tex"), "\n".join(tex) + "\n")
    atomic_write_text(ctx.p("results", "summary.md"), "\n".join(md) + "\n")
    log.info("values.tex (%d macros) and summary.md written", len(seen))
