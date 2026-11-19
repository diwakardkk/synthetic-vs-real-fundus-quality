"""Additional analyses answering anticipated reviewer concerns (no GPU; reads existing results/predictions).

A  pooled (meta-analytic) validity across datasets: Spearman rho of each predictor and RealCal - B1 difference
B  precision statement: largest effect compatible with the data (upper 95% bound of each pooled estimate)
C  sample size: how many LQ images a benchmark needs before its loss ranking is reliable (noise ceiling 0.8/0.9)
D  why RealCal fails: realism (real vs synthetic separability), sparsity of weights, per-group synthetic vs real
E  provenance of the freeze: hash of frozen weights vs lock, lock time vs first external-data artefact
F  deferral / quality-gate simulation: DeepDRiD (quality label vs model-uncertainty deferral of missed
   referable cases) and DDR (detector / uncertainty triage of ungradable images)

Usage (from realcal_code, venv active):
    python scripts/extras.py                 # everything; F-DeepDRiD needs the prediction files
    python scripts/extras.py --out /path/to/realcal_runs   # alternative output root (results/ read from there)

Writes <out>/results/extras/{extras.json, extras_summary.md, *.csv} and <out>/figures/supp_{meta,deferral}.{pdf,png}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("extras")
Z = 1.959963984540054
METHODS = ["B1_uniform", "B2_published", "RealCal_L1a_group", "RealCal_L1a_std", "RealCal_L1b_condition",
           "RealCal_L2_distribution"]
VALIDITY = {"deepdrid": "exp3_validity_internal.csv", "fives": "exp4_fives_validity.csv",
            "ddr_repl": "exp4_ddr_repl_validity.csv"}


# ----------------------------------------------------------------------------- helpers

def rj(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def rc(p: Path):
    return pd.read_csv(p) if p.exists() else None


def ci(v):
    if isinstance(v, str):
        try:
            a, b = json.loads(v)
            return float(a), float(b)
        except (ValueError, TypeError):
            return np.nan, np.nan
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    return np.nan, np.nan


def spearman(x, y) -> float:
    s = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    return float(s["x"].rank().corr(s["y"].rank())) if len(s) >= 3 else np.nan


def pool(y, se) -> dict:
    """Inverse-variance fixed effect + DerSimonian-Laird random effects."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, v = y[ok], se[ok] ** 2
    k = len(y)
    if k == 0:
        return {"k": 0}
    w = 1 / v
    fe = float((w * y).sum() / w.sum())
    fe_se = float(math.sqrt(1 / w.sum()))
    q = float((w * (y - fe) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    ws = 1 / (v + tau2)
    re_ = float((ws * y).sum() / ws.sum())
    re_se = float(math.sqrt(1 / ws.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    return {"k": k, "fixed": fe, "fixed_se": fe_se, "random": re_, "random_se": re_se, "tau2": tau2, "Q": q, "I2": i2}


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "--" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    cols = [str(c) for c in df.columns]
    rows = [[f(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


# ----------------------------------------------------------------------------- A + B

def meta_analysis(R: Path) -> tuple[dict, pd.DataFrame]:
    tabs = {k: rc(R / f) for k, f in VALIDITY.items()}
    tabs = {k: t.set_index("method") for k, t in tabs.items() if t is not None and len(t)}
    sets = {"independent (DeepDRiD + FIVES)": [d for d in ("deepdrid", "fives") if d in tabs],
            "all three (incl. DDR replication)": [d for d in ("deepdrid", "fives", "ddr_repl") if d in tabs]}
    rows, out = [], {"note": "DDR replication is evaluated on DeepDRiD test images, so it is not independent of the "
                             "internal analysis; the 'independent' pool is primary."}
    for m in METHODS:
        for quantity in ("rho", "diff_vs_B1"):
            if quantity == "diff_vs_B1" and m == "B1_uniform":
                continue
            per = {}
            for d, t in tabs.items():
                if m not in t.index:
                    continue
                if quantity == "rho":
                    r = float(t.loc[m, "spearman"])
                    lo, hi = ci(t.loc[m, "spearman_ci"])
                    zl, zh = np.arctanh(np.clip([lo, hi], -0.999, 0.999))
                    per[d] = (float(np.arctanh(np.clip(r, -0.999, 0.999))), float((zh - zl) / (2 * Z)), r, lo, hi)
                else:
                    if "diff_vs_B1" not in t or not np.isfinite(t.loc[m, "diff_vs_B1"]):
                        continue
                    r, lo, hi = (float(t.loc[m, c]) for c in ("diff_vs_B1", "diff_vs_B1_lo", "diff_vs_B1_hi"))
                    per[d] = (r, (hi - lo) / (2 * Z), r, lo, hi)
            for pname, dss in sets.items():
                use = [d for d in dss if d in per]
                if not use:
                    continue
                p = pool([per[d][0] for d in use], [per[d][1] for d in use])
                est, se = p["random"], p["random_se"]
                lo, hi = est - Z * se, est + Z * se
                if quantity == "rho":
                    est, lo, hi = (float(np.tanh(x)) for x in (est, lo, hi))
                rows.append({"method": m, "quantity": quantity, "pool": pname, "k": p["k"], "estimate": est,
                             "ci_lo": lo, "ci_hi": hi, "I2": p["I2"],
                             "per_dataset": "; ".join(f"{d} {per[d][2]:.2f} ({per[d][3]:.2f} to {per[d][4]:.2f})" for d in use)})
    t = pd.DataFrame(rows)
    if len(t):
        prim = t[t["pool"].str.startswith("independent")]
        out["pooled"] = t.to_dict(orient="records")
        b = {}
        for _, r in prim.iterrows():
            if r["quantity"] == "diff_vs_B1":
                b[r["method"]] = {"pooled_diff": r["estimate"], "max_compatible_gain_over_B1": r["ci_hi"],
                                  "max_compatible_loss_vs_B1": r["ci_lo"]}
            elif r["method"] == "B1_uniform":
                b["B1_uniform_validity"] = {"pooled_rho": r["estimate"], "max_compatible_rho": r["ci_hi"]}
        out["precision"] = b
    return out, t


# ----------------------------------------------------------------------------- C

def sample_size(R: Path) -> tuple[dict, pd.DataFrame]:
    via = rj(R / "viability.json") or {}
    fiv = rj(R / "exp4_fives_summary.json") or {}
    sets = [("deepdrid", rj(R / "noise_ceiling.json"), (via.get("strata", {}).get("lq_any", {}) or {}).get("n")),
            ("fives", fiv.get("noise_ceiling"), fiv.get("n_lq_test"))]
    rows = []
    for ds, nc, n_lq in sets:
        if not nc:
            continue
        r_p, r_s, k_s = nc.get("reliability_patient"), nc.get("reliability_seed"), nc.get("n_seeds", 3)
        r1 = nc.get("seed_pairwise_spearman")
        for target in (0.8, 0.9):
            for seeds_lab, rs in (("current seeds", r_s), ("noise-free training (many seeds)", 1.0)):
                T = target ** 2 + 1 - rs          # patient reliability needed so that sqrt(rel_p + rel_s - 1) = target
                if not (np.isfinite(r_p) and r_p > 0) or T >= 1:
                    k = np.inf
                elif T <= r_p:
                    k = 1.0
                else:
                    k = T * (1 - r_p) / (r_p * (1 - T))
                rows.append({"dataset": ds, "target_ceiling": target, "training_noise": seeds_lab,
                             "current_n_lq": n_lq, "current_ceiling": nc.get("ceiling_rho_combined"),
                             "patient_reliability_needed": T, "size_multiplier": k,
                             "lq_images_needed": (int(math.ceil(k * n_lq)) if (n_lq and np.isfinite(k)) else np.nan)})
        # seeds needed for seed reliability 0.95
        if r1 and 0 < r1 < 1:
            rows.append({"dataset": ds, "target_ceiling": np.nan, "training_noise": "seeds for seed-reliability 0.95",
                         "current_n_lq": n_lq, "current_ceiling": nc.get("ceiling_rho_combined"),
                         "patient_reliability_needed": np.nan,
                         "size_multiplier": 0.95 * (1 - r1) / (r1 * 0.05), "lq_images_needed": np.nan})
    t = pd.DataFrame(rows)
    return {"sample_size": t.to_dict(orient="records"),
            "assumption": "Spearman-Brown on the patient split-half reliability; seed and patient noise independent"}, t


# ----------------------------------------------------------------------------- D

def why_realcal(R: Path) -> dict:
    e3 = rj(R / "exp3_summary.json") or {}
    ms = rj(R / "exp4_mshf_summary.json") or {}
    out: dict = {}
    gw = e3.get("group_weights") or {}
    out["group_weights"] = gw
    out["n_nonzero_groups"] = int(sum(1 for v in gw.values() if v > 1e-6))
    out["realism_auroc_deepdrid"] = {k: v.get("auroc") for k, v in (e3.get("realism") or {}).items()}
    out["realism_auroc_mshf"] = {cam: {k: v.get("auroc") for k, v in d.items()} for cam, d in (ms.get("realism") or {}).items()}
    s = rc(R / "exp2_scores.csv")
    if s is not None:
        groups = [c for c in s.columns if c.startswith("group_")]
        per = {}
        for g in groups:
            per[g.replace("group_", "")] = {q: spearman(s[g], s[f"delta_real_{q}"])
                                            for q in ("any", "artifact", "clarity", "field") if f"delta_real_{q}" in s}
        out["spearman_group_synthetic_vs_real"] = per
        out["matched_pairs"] = {"blur vs clarity": per.get("blur", {}).get("clarity"),
                                "field vs field": per.get("field", {}).get("field"),
                                "illumination vs artifact": per.get("illumination", {}).get("artifact")}
        out["median_loss_synthetic_uniform"] = float(s["uniform"].median())
        out["median_loss_real_any"] = float(s["delta_real_any"].median())
    return out


# ----------------------------------------------------------------------------- E

def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def provenance(out_root: Path) -> dict:
    pre = out_root / "prereg"
    lock, fw = rj(pre / "stage2_lock.json"), pre / "frozen_weights.json"
    res: dict = {"lock_present": lock is not None, "frozen_weights_present": fw.exists()}
    if not lock or not fw.exists():
        return res
    cur = _sha(fw)
    res.update({"lock_created": lock.get("created"), "frozen_sha256_in_lock": lock.get("frozen_sha256"),
                "frozen_sha256_now": cur, "hash_matches": cur == lock.get("frozen_sha256")})
    f = rj(fw) or {}
    res["code_hash"], res["config_hash"] = f.get("code_hash"), f.get("config_hash")
    t_lock = datetime.strptime(lock["created"], "%Y-%m-%d %H:%M:%S").timestamp()
    first = {}
    for ds in ("fives", "mshf", "ddr"):
        cands = []
        for sub in (f"predictions/{ds}", f"runs/train/{ds}", f"cache/features/{ds}"):
            d = out_root / sub
            if d.exists():
                cands += [p.stat().st_mtime for p in d.rglob("*") if p.is_file()]
        for lg in ("logs/ext.log", "logs/final.log"):
            p = out_root / lg
            if p.exists():
                for line in p.read_text(errors="ignore").splitlines():
                    if f"/{ds}/" in line or f" {ds}" in line:
                        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
                        if m:
                            cands.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
                            break
        if cands:
            t0 = min(cands)
            first[ds] = {"first_artefact": datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                         "after_lock": bool(t0 >= t_lock - 1)}
    res["external_first_artefacts"] = first
    res["all_external_after_lock"] = bool(first) and all(v["after_lock"] for v in first.values())
    res["caveat"] = ("file modification times can be reset by later re-runs; the pipeline additionally refuses every "
                     "external stage until the lock exists and matches the frozen-weights hash (realcal/prereg.py)")
    return res


# ----------------------------------------------------------------------------- F

def deferral_deepdrid(ctx) -> tuple[dict, pd.DataFrame]:
    from realcal.eval.store import load_predictions

    df = load_predictions(ctx, "deepdrid", conditions=["clean", "real"])
    if df.empty:
        return {"error": "no DeepDRiD predictions"}, pd.DataFrame()
    df = df[df["label"].notna()].copy()
    cond = df["condition"].astype(str)
    df = df[(cond == "clean") | ((cond == "real") & df["lq_any"].astype(bool))]
    df["is_lq"] = df["condition"].astype(str) == "real"
    budgets = [0.1, 0.2, 0.3]
    rows = []
    for (b, m, s), g in df.groupby(["backbone", "mode", "seed"]):
        y = g["label"].astype(int).to_numpy()
        pred = g["pred_at_threshold"].astype(int).to_numpy()
        conf = np.abs(g["prob"].to_numpy(float) - g["threshold"].to_numpy(float))
        lq = g["is_lq"].to_numpy(bool)
        fn = (y == 1) & (pred == 0)
        err = y != pred
        if fn.sum() == 0:
            continue
        order = np.argsort(conf)                     # least confident first
        r = {"config": f"{b}/{m}", "seed": s, "n": len(y), "n_missed": int(fn.sum()),
             "lq_share_images": lq.mean(), "lq_share_missed": fn[lq].sum() / fn.sum(),
             "lq_share_errors": err[lq].sum() / max(1, err.sum()),
             "quality_gate_budget": lq.mean(), "quality_gate_missed_caught": fn[lq].sum() / fn.sum()}
        kq = int(round(lq.mean() * len(y)))
        r["uncertainty_at_quality_budget_missed_caught"] = fn[order[:kq]].sum() / fn.sum()
        for bud in budgets:
            k = int(round(bud * len(y)))
            r[f"uncertainty_{int(bud * 100)}pct_missed_caught"] = fn[order[:k]].sum() / fn.sum()
            # quality gate restricted to the same budget: least-confident LQ images first
            lq_idx = np.flatnonzero(lq)
            lq_idx = lq_idx[np.argsort(conf[lq_idx])][:k]
            r[f"lq_then_uncertainty_{int(bud * 100)}pct_missed_caught"] = fn[lq_idx].sum() / fn.sum()
        rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return {"error": "no rows"}, t
    t = t.groupby("config").mean(numeric_only=True).reset_index().drop(columns=["seed"])
    num = [c for c in t.columns if c not in ("config", "n", "n_missed")]
    summ = {c: {"median": float(t[c].median()), "q1": float(t[c].quantile(0.25)), "q3": float(t[c].quantile(0.75))}
            for c in num}
    return {"deepdrid": summ, "n_configs": int(len(t)),
            "note": "fractions of each model's missed referable cases (false negatives at the frozen threshold) that a "
                    "deferral policy would route to a human grader; random deferral catches a fraction equal to the budget"}, t


def deferral_ddr(R: Path) -> tuple[dict, pd.DataFrame]:
    det = R / "exp4_ddr_detector_scores.parquet"
    ent = R / "exp4_ddr_entropy_rows.parquet"
    if not det.exists() or not ent.exists():
        return {"error": "DDR detector/entropy files missing"}, pd.DataFrame()
    d = pd.read_parquet(det)
    e = pd.read_parquet(ent)
    budgets = [0.05, 0.1, 0.2]
    out: dict = {"prevalence_ungradable": float(d["ungradable_f"].mean())}
    for col in ("score_uniform", "score_realcal"):
        for bud in budgets:
            thr = d[col].quantile(1 - bud)
            flag = d[col] >= thr
            out[f"{col}_{int(bud * 100)}pct_sensitivity_ungradable"] = float(flag[d["ungradable_f"] == 1].mean())
    rows = []
    flags = {col: {bud: set(d.loc[d[col] >= d[col].quantile(1 - bud), "image_id"]) for bud in budgets}
             for col in ("score_uniform", "score_realcal")}
    n_grad_total = int((d["ungradable_f"] == 0).sum())
    for (b, m), g in e.groupby(["backbone", "mode"]):
        ung = g["ungradable"].astype(bool).to_numpy()
        conf_ung = g.loc[ung & (g["maxp"] > 0.9).to_numpy(), "image_id"]
        r = {"config": f"{b}/{m}", "n_ungradable": int(ung.sum()), "n_confident_ungradable": int(len(conf_ung))}
        # gradable images are subsampled in the entropy rows: weight them back to the full DDR population
        n_grad_s = max(1, int((~ung).sum()))
        w = np.where(ung, 1.0, n_grad_total / n_grad_s)
        order = np.argsort(g["maxp"].to_numpy())
        cw = np.cumsum(w[order]) / w.sum()
        for bud in budgets:
            k = int(np.searchsorted(cw, bud, side="right"))
            r[f"uncertainty_{int(bud * 100)}pct_sensitivity_ungradable"] = ung[order[:k]].sum() / max(1, ung.sum())
            for col in ("score_uniform", "score_realcal"):
                r[f"{col}_{int(bud * 100)}pct_catches_confident_ungradable"] = (
                    conf_ung.isin(flags[col][bud]).mean() if len(conf_ung) else np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    out["per_config_median"] = {c: float(t[c].median()) for c in t.columns if c not in ("config",)}
    out["note"] = ("budgets are fractions of all DDR images flagged; uncertainty = lowest max-probability images of each "
                   "DeepDRiD model (gradable images are subsampled in the entropy rows and re-weighted to the full DDR population)")
    return out, t


# ----------------------------------------------------------------------------- figures

def figures(F: Path, meta_t: pd.DataFrame, defer_t: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    F.mkdir(parents=True, exist_ok=True)
    MM = 1 / 25.4
    if meta_t is not None and len(meta_t):
        t = meta_t[meta_t["pool"].str.startswith("independent")]
        fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 55 * MM))
        for ax, (q, lab) in zip(axes, [("rho", "pooled Spearman rho with real loss"),
                                       ("diff_vs_B1", "pooled difference in rho vs uniform (B1)")]):
            s = t[t["quantity"] == q].reset_index(drop=True)
            y = np.arange(len(s))[::-1]
            ax.errorbar(s["estimate"], y, xerr=[s["estimate"] - s["ci_lo"], s["ci_hi"] - s["estimate"]], fmt="o",
                        color="#0072B2", ms=3, capsize=1.5, lw=0.8)
            ax.axvline(0, color="grey", lw=0.5, ls="--")
            ax.set_yticks(y, s["method"])
            ax.set_xlabel(lab)
        axes[0].set_title("a  DeepDRiD + FIVES (random effects)", loc="left", fontweight="bold")
        axes[1].set_title("b", loc="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(F / "supp_meta.pdf", bbox_inches="tight")
        fig.savefig(F / "supp_meta.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    if defer_t is not None and len(defer_t):
        cols = [("quality_gate_missed_caught", "quality label\n(all LQ)"),
                ("uncertainty_at_quality_budget_missed_caught", "uncertainty\n(same budget)"),
                ("uncertainty_10pct_missed_caught", "uncertainty\n10%"),
                ("lq_then_uncertainty_10pct_missed_caught", "LQ, least\nconfident 10%"),
                ("uncertainty_20pct_missed_caught", "uncertainty\n20%"),
                ("lq_then_uncertainty_20pct_missed_caught", "LQ, least\nconfident 20%")]
        cols = [(c, l) for c, l in cols if c in defer_t]
        budget = [defer_t["quality_gate_budget"].median(), defer_t["quality_gate_budget"].median(), 0.1, 0.1, 0.2, 0.2]
        fig, ax = plt.subplots(figsize=(120 * MM, 60 * MM))
        for i, (c, _) in enumerate(cols):
            v = 100 * defer_t[c].to_numpy()
            ax.scatter(i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=6, color="#0072B2", alpha=0.7, lw=0)
            ax.hlines(np.median(v), i - 0.3, i + 0.3, color="k", lw=1.2)
            ax.hlines(100 * budget[i], i - 0.35, i + 0.35, color="#D55E00", lw=0.8, ls=":")
        ax.set_xticks(range(len(cols)), [l for _, l in cols], fontsize=6)
        ax.set_ylabel("% of missed referable cases deferred")
        ax.set_title("DeepDRiD: deferral policies (orange = random deferral)", fontsize=7, loc="left")
        fig.tight_layout()
        fig.savefig(F / "supp_deferral.pdf", bbox_inches="tight")
        fig.savefig(F / "supp_deferral.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output root (default: paths.out from the config)")
    ap.add_argument("--no-predictions", action="store_true", help="skip F-DeepDRiD (needs prediction files)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctx = None
    if a.out:
        out_root = Path(a.out)
    else:
        from realcal.config import Ctx, load_config
        ctx = Ctx(load_config())
        out_root = ctx.out
    R = out_root / "results"
    E = R / "extras"
    E.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    meta, meta_t = meta_analysis(R)
    res["A_B_meta_analysis"] = meta
    meta_t.to_csv(E / "meta_analysis.csv", index=False)
    ss, ss_t = sample_size(R)
    res["C_sample_size"] = ss
    ss_t.to_csv(E / "sample_size.csv", index=False)
    res["D_why_realcal"] = why_realcal(R)
    res["E_provenance"] = provenance(out_root)
    dd, dd_t = deferral_ddr(R)
    res["F_deferral_ddr"] = dd
    dd_t.to_csv(E / "deferral_ddr.csv", index=False)
    dp_t = pd.DataFrame()
    if not a.no_predictions:
        try:
            if ctx is None:
                from realcal.config import Ctx, load_config
                ctx = Ctx(load_config())
                ctx.out = out_root  # type: ignore[misc]
            dp, dp_t = deferral_deepdrid(ctx)
            res["F_deferral_deepdrid"] = dp
            dp_t.to_csv(E / "deferral_deepdrid.csv", index=False)
        except Exception as e:  # noqa: BLE001
            log.warning("F-DeepDRiD skipped: %s", e)
            res["F_deferral_deepdrid"] = {"error": str(e)}
    (E / "extras.json").write_text(json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else str(o)))
    try:
        figures(out_root / "figures", meta_t, dp_t)
    except Exception as e:  # noqa: BLE001
        log.warning("figures skipped: %s", e)

    # human-readable summary
    L = ["# Extra analyses (A-F)", ""]
    if len(meta_t):
        L += ["## A/B. Pooled validity (random effects)", "", md(meta_t.drop(columns=["per_dataset"])), "",
              "Per-dataset inputs:", ""] + [f"- {r.method} / {r.quantity} / {r.pool}: {r.per_dataset}"
                                           for r in meta_t.itertuples()] + [""]
        L += ["Precision (primary pool):", "```", json.dumps(meta.get("precision", {}), indent=1), "```", ""]
    if len(ss_t):
        L += ["## C. LQ images needed for a reliable benchmark", "", md(ss_t), ""]
    L += ["## D. Why RealCal does not help", "```", json.dumps(res["D_why_realcal"], indent=1, default=str), "```", ""]
    L += ["## E. Provenance of the freeze", "```", json.dumps(res["E_provenance"], indent=1, default=str), "```", ""]
    L += ["## F. Deferral", "", "DDR:", "```", json.dumps(dd, indent=1, default=str), "```", ""]
    if len(dp_t):
        L += ["DeepDRiD (per configuration, mean over seeds):", "", md(dp_t), "",
              "```", json.dumps(res["F_deferral_deepdrid"], indent=1, default=str), "```"]
    (E / "extras_summary.md").write_text("\n".join(L))
    log.info("extras written to %s", E)


if __name__ == "__main__":
    main()
