"""Paper figures -> ``figures/<name>.pdf`` (vector) and ``.png`` (300 dpi).

Single column 88 mm, double column 180 mm, 7-8 pt sans-serif, colour-blind-safe palette
(Okabe-Ito; family = colour, adaptation mode = marker shape).
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from realcal.config import Ctx  # noqa: E402
from realcal.utils.io import read_csv_safe, read_json  # noqa: E402

log = logging.getLogger(__name__)

FAMILY_COLOURS = {"retina": "#0072B2", "medical": "#009E73", "general": "#E69F00", "cnn": "#CC79A7", "analysis": "#999999"}
MODE_MARKERS = {"lp": "o", "lora": "s", "ft": "^"}
MM = 1 / 25.4


def style(ctx: Ctx) -> None:
    fs = float(ctx.cfg.figures.font_size)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": fs, "axes.titlesize": fs + 0.5, "axes.labelsize": fs, "xtick.labelsize": fs - 0.5,
                         "ytick.labelsize": fs - 0.5, "legend.fontsize": fs - 0.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.0})


def save(ctx: Ctx, fig, name: str) -> None:
    out = ctx.p("figures", f"{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=int(ctx.cfg.figures.dpi), bbox_inches="tight")
    plt.close(fig)


def width(ctx: Ctx, double: bool) -> float:
    return (float(ctx.cfg.figures.double_col_mm) if double else float(ctx.cfg.figures.single_col_mm)) * MM


def _res(ctx, name):
    p = ctx.out / "results" / name
    if not p.exists():
        return None
    if name.endswith(".csv"):
        return read_csv_safe(p)
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    return read_json(p)


def scatter_configs(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for _, r in df.iterrows():
        ax.scatter(r[x], r[y], c=FAMILY_COLOURS.get(r["family"], "k"), marker=MODE_MARKERS.get(r["mode"], "o"),
                   s=18, edgecolor="white", linewidth=0.4, zorder=3)


def legend_family_mode(ax) -> None:
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=c, marker="o", ls="", ms=4, label=f) for f, c in FAMILY_COLOURS.items() if f != "analysis"]
    h += [Line2D([], [], color="grey", marker=m, ls="", ms=4, label=k) for k, m in MODE_MARKERS.items()]
    ax.legend(handles=h, frameon=False, loc="best", ncol=2, handletextpad=0.2, columnspacing=0.6)


def fit_band(ax, x, y) -> None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    xs = np.linspace(x.min(), x.max(), 50)
    Xs = np.c_[np.ones_like(xs), xs]
    resid = y - X @ beta
    s2 = resid @ resid / max(1, len(x) - 2)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))
    ax.plot(xs, Xs @ beta, color="k", lw=0.8)
    ax.fill_between(xs, Xs @ beta - 1.96 * se, Xs @ beta + 1.96 * se, color="grey", alpha=0.2, lw=0)


def fig2(ctx: Ctx) -> None:
    t = _res(ctx, "exp1_real_loss.csv")
    pairs = _res(ctx, "exp1_within_eye_pairs.csv")
    if t is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(width(ctx, True), 55 * MM), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for ax, (pref, lab) in zip(axes[:2], [("delta_real_", "AUROC loss (HQ - LQ)"), ("delta_real_sens_", "Sensitivity loss (HQ - LQ)")]):
        qs = ["any", "artifact", "clarity", "field"]
        qlab = ["any", "artefact", "clarity", "field"]
        for i, q in enumerate(qs):
            col = f"{pref}{q}"
            if col not in t:
                continue
            v = t[col].to_numpy()
            jit = np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(i + jit, v, c=[FAMILY_COLOURS.get(fm, "k") for fm in t["family"]], s=8, alpha=0.8, lw=0)
            ax.hlines(np.nanmedian(v), i - 0.3, i + 0.3, color="k", lw=1.2)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xticks(range(4), qlab)
        ax.set_ylabel(lab)
    axes[0].set_title("a", loc="left", fontweight="bold")
    axes[1].set_title("b", loc="left", fontweight="bold")
    if pairs is not None and len(pairs):
        p = pairs.sort_values("config")
        y = np.arange(len(p))
        axes[2].barh(y, p["hq_only_correct"], color="#0072B2", label="correct on HQ only")
        axes[2].barh(y, -p["lq_only_correct"], color="#E69F00", label="correct on LQ only")
        axes[2].set_yticks(y, p["config"], fontsize=5)
        axes[2].axvline(0, color="k", lw=0.5)
        axes[2].set_xlabel("discordant within-eye pairs")
        axes[2].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.5)
        axes[2].set_title("c", loc="left", fontweight="bold", pad=14)
    fig.tight_layout()
    save(ctx, fig, "fig2_real_loss")


def fig3(ctx: Ctx) -> None:
    s = _res(ctx, "exp2_scores.csv")
    h1 = _res(ctx, "h1.json")
    if s is None:
        return
    fig, ax = plt.subplots(figsize=(width(ctx, False), 70 * MM))
    scatter_configs(ax, s, "uniform", "delta_real_any")
    fit_band(ax, s["uniform"], s["delta_real_any"])
    ax.set_xlabel("Uniform synthetic AUROC loss (mean of 75 conditions)")
    ax.set_ylabel("Real AUROC loss (HQ - LQ)")
    if h1:
        lo, hi = h1.get("rho_ci", [np.nan, np.nan])
        ax.set_title(f"Spearman rho = {h1.get('rho', np.nan):.2f} ({lo:.2f} to {hi:.2f}); {h1.get('zone')}",
                     fontsize=6.5, loc="left")
        ins = ax.inset_axes([0.62, 0.06, 0.35, 0.08])
        ins.axvspan(-1, 0.5, color="#D55E00", alpha=0.25)
        ins.axvspan(0.5, 0.7, color="grey", alpha=0.25)
        ins.axvspan(0.7, 1, color="#009E73", alpha=0.25)
        ins.errorbar([h1.get("rho", np.nan)], [0], xerr=[[h1.get("rho", 0) - lo], [hi - h1.get("rho", 0)]], fmt="o", color="k", ms=2, lw=0.8)
        ins.set_xlim(-0.5, 1)
        ins.set_yticks([])
        ins.set_xticks([-0.5, 0, 0.5, 1])
        ins.tick_params(labelsize=5)
    legend_family_mode(ax)
    save(ctx, fig, "fig3_h1_uniform_vs_real")


def fig4(ctx: Ctx) -> None:
    w = _res(ctx, "exp3_group_weights.csv")
    c = _res(ctx, "exp3_condition_weights.csv")
    sc = _res(ctx, "exp3_scores.csv")
    if w is None or sc is None:
        return
    fig = plt.figure(figsize=(width(ctx, True), 65 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.2, 1])
    a = fig.add_subplot(gs[0])
    err = np.vstack([w["weight"] - w["ci_lo"], w["ci_hi"] - w["weight"]]).clip(min=0)
    a.bar(w["group"], w["weight"], yerr=err, color="#0072B2", capsize=1.5, error_kw={"lw": 0.6})
    a.set_ylabel("Level-1a weight")
    a.tick_params(axis="x", rotation=60)
    a.set_title("a", loc="left", fontweight="bold")
    h = fig.add_subplot(gs[1])
    if c is not None and len(c):
        piv = c.pivot_table(index="corruption", columns="severity", values="weight", aggfunc="sum").fillna(0)
        im = h.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
        h.set_yticks(range(len(piv)), piv.index, fontsize=5)
        h.set_xticks(range(piv.shape[1]), [f"S{k}" for k in piv.columns])
        fig.colorbar(im, ax=h, fraction=0.05, label="Level-1b weight")
    b = fig.add_subplot(gs[2])
    scatter_configs(b, sc, "RealCal_L1a_group", "delta_real_any")
    lim = [np.nanmin(sc[["RealCal_L1a_group", "delta_real_any"]].to_numpy()), np.nanmax(sc[["RealCal_L1a_group", "delta_real_any"]].to_numpy())]
    b.plot(lim, lim, color="grey", ls="--", lw=0.6)
    b.set_xlabel("RealCal score (LOBO)")
    b.set_ylabel("Real AUROC loss")
    b.set_title("b", loc="left", fontweight="bold")
    fig.tight_layout()
    save(ctx, fig, "supp_realcal_weights")   # RealCal is a secondary method in the evaluation framing


def _panel(ax, letter: str) -> None:
    ax.set_title(letter, loc="left", fontweight="bold")


def fig5(ctx: Ctx) -> None:
    """External quality datasets: MSHF and DDR detectors, and over-confidence on DDR ungradable images."""
    from sklearn.metrics import roc_auc_score, roc_curve

    ms = _res(ctx, "exp4_mshf_detector_scores.parquet")
    dd = _res(ctx, "exp4_ddr_detector_scores.parquet")
    ent = _res(ctx, "exp4_ddr_entropy_rows.parquet")
    ent_t = _res(ctx, "exp4_ddr_entropy.csv")
    if ms is None and dd is None and ent is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(width(ctx, True), 120 * MM))
    (a, b), (c, d) = axes
    cols = [("score_uniform", "--", "uniform synthetic"), ("score_realcal", "-", "RealCal")]
    if ms is not None:
        cam_col = {"portable": "#D55E00", "tabletop": "#0072B2"}
        for cam, g in ms.groupby("camera"):
            y = g["lq_any"].astype(int)
            for col, ls, lab in cols:
                fpr, tpr, _ = roc_curve(y, g[col])
                a.plot(fpr, tpr, ls=ls, color=cam_col.get(str(cam), "k"),
                       label=f"{cam}, {lab} ({roc_auc_score(y, g[col]):.2f})")
        a.plot([0, 1], [0, 1], color="grey", lw=0.5)
        a.set_xlabel("1 - specificity")
        a.set_ylabel("sensitivity")
        a.legend(frameon=False, fontsize=5, loc="lower right")
        a.text(0.02, 0.98, "MSHF: low-quality detection", transform=a.transAxes, va="top", fontsize=6)
        _panel(a, "a")
    if dd is not None:
        y = dd["ungradable_f"].astype(int)
        for col, ls, lab in cols:
            fpr, tpr, _ = roc_curve(y, dd[col])
            b.plot(fpr, tpr, ls=ls, color="k", label=f"{lab} ({roc_auc_score(y, dd[col]):.2f})")
        b.plot([0, 1], [0, 1], color="grey", lw=0.5)
        b.set_xlabel("1 - specificity")
        b.set_ylabel("sensitivity")
        b.legend(frameon=False, fontsize=6, loc="lower right")
        b.text(0.02, 0.98, "DDR: ungradable detection", transform=b.transAxes, va="top", fontsize=6)
        _panel(b, "b")
    if ent is not None:
        for ung, col in [(True, "#D55E00"), (False, "#0072B2")]:
            v = ent[ent["ungradable"].astype(bool) == ung]["maxp"]
            c.hist(v, bins=25, range=(0.5, 1), alpha=0.6, color=col, density=True,
                   label="ungradable" if ung else "gradable")
        c.axvline(0.9, color="k", lw=0.6, ls=":")
        c.set_xlabel("maximum class probability (DeepDRiD models on DDR)")
        c.set_ylabel("density")
        c.legend(frameon=False)
        _panel(c, "c")
    if ent_t is not None and len(ent_t):
        t = ent_t.sort_values("share_ungradable_confident")
        yy = np.arange(len(t))
        d.barh(yy, 100 * t["share_ungradable_confident"], color="#D55E00", height=0.7, label="ungradable")
        d.scatter(100 * t["share_gradable_confident"], yy, color="#0072B2", s=6, zorder=3, label="gradable")
        d.set_yticks(yy, t["config"], fontsize=4.5)
        d.set_xlabel("% images predicted with probability > 0.9")
        d.legend(frameon=False, fontsize=5.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
        _panel(d, "d")
    fig.tight_layout()
    save(ctx, fig, "fig5_external")


# ----------------------------------------------------------------------------- evaluation-paper figures

METHOD_LABELS = {"B1_uniform": "Uniform synthetic (B1)", "B2_published": "Published suite (B2)",
                 "B3_best_single": "Best single corruption (B3)", "B4_clean_perf": "Clean AUROC (B4)",
                 "B5_balanced": "Balanced groups (B5)", "RealCal_L1a_group": "RealCal group weights",
                 "RealCal_L1b_condition": "RealCal condition weights", "RealCal_L2_distribution": "RealCal distribution",
                 "RealCal_L1a_std": "RealCal standardised"}


def _ci(v):
    import json

    if isinstance(v, str):
        try:
            lo, hi = json.loads(v)
            return float(lo), float(hi)
        except (ValueError, TypeError):
            return np.nan, np.nan
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return float(v[0]), float(v[1])
    return np.nan, np.nan


def _spearman(x, y) -> float:
    s = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    if len(s) < 3:
        return np.nan
    return float(s["x"].rank().corr(s["y"].rank()))


def fig_forest(ctx: Ctx) -> None:
    """Spearman rho (95% CI) of every predictor against the real loss, per dataset, with noise ceilings."""
    sets = [("DeepDRiD (internal, LOBO)", "exp3_validity_internal.csv", _res(ctx, "noise_ceiling.json")),
            ("FIVES (external)", "exp4_fives_validity.csv", (_res(ctx, "exp4_fives_summary.json") or {}).get("noise_ceiling")),
            ("Replication: DDR-trained", "exp4_ddr_repl_validity.csv", None)]
    tabs = [(name, _res(ctx, f), nc) for name, f, nc in sets]
    tabs = [x for x in tabs if x[1] is not None and len(x[1])]
    if not tabs:
        return
    methods = [m for m in METHOD_LABELS if any(m in set(t["method"]) for _, t, _ in tabs)]
    fig, axes = plt.subplots(1, len(tabs), figsize=(width(ctx, True), (25 + 6 * len(methods)) * MM), sharey=True,
                             squeeze=False)
    yy = np.arange(len(methods))[::-1]
    for k, (ax, (name, t, nc)) in enumerate(zip(axes[0], tabs)):
        t = t.set_index("method")
        for y, m in zip(yy, methods):
            if m not in t.index:
                continue
            r = float(t.loc[m, "spearman"])
            lo, hi = _ci(t.loc[m, "spearman_ci"])
            col = "#0072B2" if m.startswith("RealCal") else ("#009E73" if m == "B4_clean_perf" else "#555555")
            ax.errorbar([r], [y], xerr=[[max(0, r - lo)], [max(0, hi - r)]], fmt="o", color=col, ms=3, lw=0.8, capsize=1.5)
        ax.axvline(0, color="grey", lw=0.5, ls="--")
        if nc and np.isfinite(nc.get("ceiling_rho_combined", np.nan)):
            c = float(nc["ceiling_rho_combined"])
            ax.axvspan(c, 1.0, color="#D55E00", alpha=0.10, lw=0)
            ax.axvline(c, color="#D55E00", lw=0.8)
            ax.text(c, yy.max() + 0.6, f" noise ceiling {c:.2f}", color="#D55E00", fontsize=5.5, va="bottom")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.7, len(methods) + 0.3)
        ax.set_xlabel("Spearman rho with real AUROC loss")
        ax.set_title(f"{'abc'[k]}   {name}", loc="left", fontweight="bold", pad=12)
    axes[0, 0].set_yticks(yy, [METHOD_LABELS[m] for m in methods])
    fig.tight_layout()
    save(ctx, fig, "fig_forest_validity")


def fig_clean_vs_loss(ctx: Ctx) -> None:
    """Clean (HQ) AUROC vs real AUROC loss: the best clean models lose the most."""
    sets = [("DeepDRiD", _res(ctx, "exp2_scores.csv")), ("FIVES", _res(ctx, "exp4_fives_scores.csv")),
            ("DDR-trained, tested on DeepDRiD", _res(ctx, "exp4_ddr_repl_scores.csv"))]
    from matplotlib.lines import Line2D
    sets = [(n, t) for n, t in sets if t is not None and len(t) and "auroc_clean" in t]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(width(ctx, True), 60 * MM), squeeze=False)
    for k, (ax, (name, t)) in enumerate(zip(axes[0], sets)):
        scatter_configs(ax, t, "auroc_clean", "delta_real_any")
        fit_band(ax, t["auroc_clean"], t["delta_real_any"])
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.text(0.03, 0.97, f"{name}\nSpearman rho = {_spearman(t['auroc_clean'], t['delta_real_any']):.2f}",
                transform=ax.transAxes, va="top", fontsize=6)
        ax.set_xlabel("AUROC on high-quality images")
        if k == 0:
            ax.set_ylabel("real AUROC loss (HQ - LQ)")
        _panel(ax, "abc"[k])
    h = [Line2D([], [], color=c, marker="o", ls="", ms=4, label=f) for f, c in FAMILY_COLOURS.items() if f != "analysis"]
    h += [Line2D([], [], color="grey", marker=m, ls="", ms=4, label=k) for k, m in MODE_MARKERS.items()]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False)
    save(ctx, fig, "fig_clean_vs_loss")


def fig_casemix(ctx: Ctx) -> None:
    """Raw vs case-mix-standardised real loss (DeepDRiD by DR grade, FIVES by disease class)."""
    sets = []
    g = _res(ctx, "exp1_grade_standardized.csv")
    if g is not None and len(g):
        sets.append(("DeepDRiD, standardised to HQ grade mix", g[g["stratum"] == "any"] if "stratum" in g else g))
    f = _res(ctx, "exp4_fives_class_standardized.csv")
    if f is not None and len(f):
        sets.append(("FIVES, standardised to HQ class mix", f[f["stratum"] == "any"] if "stratum" in f else f))
    if not sets:
        return
    n = max(len(t) for _, t in sets)
    fig, axes = plt.subplots(1, len(sets), figsize=(width(ctx, True), (20 + 3.2 * n) * MM), squeeze=False)
    for k, (ax, (name, t)) in enumerate(zip(axes[0], sets)):
        t = t.sort_values("delta_std")
        yy = np.arange(len(t))
        ax.hlines(yy, t["delta_raw"], t["delta_std"], color="grey", lw=0.6)
        ax.scatter(t["delta_raw"], yy, color="#999999", s=8, zorder=3, label="raw")
        ax.scatter(t["delta_std"], yy, color="#D55E00", s=8, zorder=3, label="standardised")
        for v, col in [(t["delta_raw"].median(), "#999999"), (t["delta_std"].median(), "#D55E00")]:
            ax.axvline(v, color=col, lw=0.6, ls=":")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_yticks(yy, t["config"], fontsize=4.5)
        ax.set_xlabel("real AUROC loss (HQ - LQ)")
        ax.text(0.5, 1.02, name, transform=ax.transAxes, ha="center", fontsize=6.5)
        ax.legend(frameon=False, fontsize=5.5, loc="best")
        _panel(ax, "ab"[k])
    fig.tight_layout()
    save(ctx, fig, "fig_casemix")


def fig_sensitivity(ctx: Ctx) -> None:
    """Augmentation and 448 px: change in synthetic loss vs change in real loss (paired per configuration)."""
    sets = [("Training augmentation", _res(ctx, "sens_aug.csv"), "aug", "aug"),
            ("448 px input", _res(ctx, "sens_s448.csv"), "s448", "448 px")]
    sets = [x for x in sets if x[1] is not None and len(x[1])]
    if not sets:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(width(ctx, True), 60 * MM), sharey=True, squeeze=False)
    for k, (ax, (name, t, suf, lab)) in enumerate(zip(axes[0], sets)):
        for j, (pre, title) in enumerate([("uniform", "synthetic loss"), ("delta_real", "real loss")]):
            a, b = t[f"{pre}_main"].to_numpy(float), t[f"{pre}_{suf}"].to_numpy(float)
            x0, x1 = 3 * j, 3 * j + 1
            for u, v in zip(a, b):
                ax.plot([x0, x1], [u, v], color="#0072B2" if j == 0 else "#D55E00", lw=0.6, alpha=0.6)
            ax.scatter(np.full(len(a), x0), a, color="grey", s=6, zorder=3)
            ax.scatter(np.full(len(b), x1), b, color="k", s=6, zorder=3)
            ax.text(x0 + 0.5, 1.0, f"{title}\nmedian change {np.nanmedian(b - a):+.3f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_xticks([0, 1, 3, 4], ["224 px" if suf == "s448" else "no aug", lab] * 2, fontsize=6)
        ax.set_xlim(-0.6, 4.6)
        if k == 0:
            ax.set_ylabel("AUROC loss")
        ax.set_title(f"{'ab'[k]}   {name} ({len(t)} configurations)", loc="left", fontweight="bold", pad=22)
    fig.tight_layout()
    save(ctx, fig, "fig_sensitivity_aug_448")


def fig_examples(ctx: Ctx, size: int = 224, n: int = 6, seed: int = 3) -> None:
    """Real low-quality images (DeepDRiD by defect type; MSHF portable) next to synthetic corruptions (severity 3)."""
    from realcal.corruptions.cache import corrupt_path, eval_hq_ids
    from realcal.corruptions.ops import run_op
    from realcal.corruptions.registry import corruption_rng, main_conditions
    from realcal.data.common import load_metadata
    from realcal.data.preprocess import mask_path, preproc_path
    from realcal.utils.io import load_gray, load_rgb

    rng = np.random.default_rng(seed)
    rows = []   # (row label, [(image, caption)])

    def pick(df, k):
        ids = [i for i in df["image_id"].astype(str) if preproc_path(ctx, ds, size, i).exists()]
        return list(rng.choice(ids, size=min(k, len(ids)), replace=False)) if ids else []

    try:
        ds = "deepdrid"
        m = load_metadata(ctx, ds)
        cells = []
        for col, lab in [("lq_artifact", "artefact"), ("lq_clarity", "clarity"), ("lq_field", "field")]:
            for i in pick(m[m[col].astype(bool)], n // 3):
                cells.append((load_rgb(preproc_path(ctx, ds, size, i)), lab))
        if cells:
            rows.append(("Real LQ\nDeepDRiD", cells))
    except Exception as e:  # noqa: BLE001
        log.warning("examples: deepdrid real skipped: %s", e)
    try:
        ds = "mshf"
        m = load_metadata(ctx, ds)
        sel = m[m["lq_any"].astype(bool) & (m["camera"].astype(str) == "portable")]
        cells = [(load_rgb(preproc_path(ctx, ds, size, i)), "portable") for i in pick(sel, n)]
        if cells:
            rows.append(("Real LQ\nMSHF", cells))
    except Exception as e:  # noqa: BLE001
        log.warning("examples: mshf real skipped: %s", e)
    try:
        ds = "deepdrid"
        suite = ctx.cfg.corruption_suite
        names = list(suite.corruptions.keys())
        names = names[:: max(1, len(names) // n)][:n]
        conds = {c.corruption: c for c in main_conditions(suite) if c.severity == 3}
        ids = eval_hq_ids(ctx, ds)
        cells = []
        for name in names:
            iid = str(rng.choice(ids))
            c = conds[name]
            p = corrupt_path(ctx, ds, size, c.name, iid)
            if p.exists():
                img = load_rgb(p)
            else:
                base = load_rgb(preproc_path(ctx, ds, size, iid))
                mask = load_gray(mask_path(ctx, ds, size, iid)) > 0
                img = run_op(c.op, base, mask, c.param, corruption_rng(iid, c), c.spatial, suite)
            cells.append((img, name.replace("_", " ")))
        if cells:
            rows.append(("Synthetic\n(severity 3)", cells))
    except Exception as e:  # noqa: BLE001
        log.warning("examples: synthetic skipped: %s", e)
    if not rows:
        return
    ncol = max(len(c) for _, c in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(width(ctx, True), len(rows) * 33 * MM), squeeze=False)
    for r, (lab, cells) in enumerate(rows):
        for j in range(ncol):
            ax = axes[r, j]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if j < len(cells):
                ax.imshow(cells[j][0])
                ax.set_title(cells[j][1], fontsize=5.5, pad=2)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
    plt.subplots_adjust(wspace=0.04, hspace=0.18)
    save(ctx, fig, "fig_examples_real_vs_synth")


def fig6(ctx: Ctx) -> None:
    t = _res(ctx, "clinical_missed.csv")
    sel = _res(ctx, "clinical_selection.csv")
    dca = _res(ctx, "clinical_dca.csv")
    if t is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(width(ctx, True), 60 * MM))
    prevs = sorted(t["prevalence"].unique())
    for j, p in enumerate(prevs):
        for k, st in enumerate(["HQ", "LQ"]):
            v = t[(t.prevalence == p) & (t.stratum == st)]["missed_per_1000"]
            x = j * 3 + k
            axes[0].scatter(np.full(len(v), x) + np.random.default_rng(x).uniform(-0.2, 0.2, len(v)), v, s=6,
                            color="#0072B2" if st == "HQ" else "#D55E00", alpha=0.7, lw=0)
    if sel is not None and len(sel):
        for how, mk in [("uniform", "x"), ("realcal", "*"), ("oracle", "D")]:
            s = sel[sel.selected_by == how]
            for _, r in s.iterrows():
                j = prevs.index(r["prevalence"])
                axes[0].scatter(j * 3 + 1, r["missed_per_1000_LQ"], marker=mk, color="k", s=20,
                                label={"uniform": "selected by uniform", "realcal": "selected by RealCal",
                                       "oracle": "oracle (best on LQ)"}[how] if j == 0 else None)
        axes[0].legend(frameon=False)
    axes[0].set_xticks([j * 3 + 0.5 for j in range(len(prevs))], [f"{100 * p:.0f}%" for p in prevs])
    axes[0].set_xlabel("prevalence (HQ blue, LQ orange)")
    axes[0].set_ylabel("missed referable cases per 1,000")
    axes[0].set_title("a", loc="left", fontweight="bold")
    if dca is not None:
        p0 = sorted(dca["prevalence"].unique())[len(dca["prevalence"].unique()) // 2]
        d = dca[dca.prevalence == p0]
        for (cfg, st), g in d.groupby(["config", "stratum"]):
            ls = "-" if st == "HQ" else ("--" if st == "LQ" else ":")
            axes[1].plot(g["pt"], g["net_benefit"], ls=ls, label=f"{cfg} {st}" if st != "all" else cfg, lw=0.8)
        axes[1].set_ylim(bottom=-0.02)
        axes[1].set_xlabel("threshold probability")
        axes[1].set_ylabel(f"net benefit (prevalence {100 * p0:.0f}%)")
        axes[1].legend(frameon=False, fontsize=5, loc="lower right", bbox_to_anchor=(1, 0.04))
        axes[1].set_title("b", loc="left", fontweight="bold")
    fig.tight_layout()
    save(ctx, fig, "fig6_clinical")


def supp(ctx: Ctx) -> None:
    ds = _res(ctx, "exp2_delta_syn.csv")
    if ds is not None:
        mat = ds.set_index("config")
        fig, ax = plt.subplots(figsize=(width(ctx, True), 90 * MM))
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma")
        ax.set_yticks(range(len(mat)), mat.index, fontsize=4.5)
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=90, fontsize=3.5)
        fig.colorbar(im, ax=ax, label="AUROC loss")
        save(ctx, fig, "supp_heatmap_delta_syn")
    pw = _res(ctx, "power_simulation.csv")
    if pw is not None:
        fig, axes = plt.subplots(1, 2, figsize=(width(ctx, True), 50 * MM))
        s = pw[pw.analysis == "single"]
        axes[0].plot(s["rho"], s["power"], marker="o")
        axes[0].axhline(0.8, color="grey", ls="--")
        axes[0].set_xlabel("true Spearman rho")
        axes[0].set_ylabel("power")
        d = pw[pw.analysis == "difference"]
        axes[1].plot(d["diff"], d["power"], marker="o")
        axes[1].axhline(0.8, color="grey", ls="--")
        axes[1].set_xlabel("true difference in rho")
        save(ctx, fig, "supp_power")
    rel = _res(ctx, "exp1_reliability.parquet")
    if rel is not None and len(rel):
        cfgs = sorted(rel["config"].unique())
        n = len(cfgs)
        ncol = 6
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(width(ctx, True), nrow * 28 * MM), squeeze=False)
        for i, c in enumerate(cfgs):
            ax = axes[i // ncol, i % ncol]
            for st, col in [("HQ", "#0072B2"), ("LQ", "#D55E00")]:
                g = rel[(rel.config == c) & (rel.stratum == st)]
                ax.plot(g["mean_pred"], g["frac_pos"], marker=".", color=col, lw=0.7, ms=2)
            ax.plot([0, 1], [0, 1], color="grey", lw=0.4)
            ax.set_title(c, fontsize=5)
            ax.tick_params(labelsize=4)
        for j in range(n, nrow * ncol):
            axes[j // ncol, j % ncol].axis("off")
        fig.tight_layout()
        save(ctx, fig, "supp_reliability")
    site = _res(ctx, "sens_subgroup_site.csv")
    if site is not None and len(site):
        fig, ax = plt.subplots(figsize=(width(ctx, False), 55 * MM))
        for i, (sname, g) in enumerate(site.groupby("site")):
            ax.scatter(np.full(len(g), i), g["delta_real"], s=6, alpha=0.7)
        ax.set_xticks(range(site["site"].nunique()), sorted(site["site"].unique()), rotation=45)
        ax.set_ylabel("real AUROC loss")
        save(ctx, fig, "supp_site_subgroups")


def make_figures(ctx: Ctx) -> None:
    style(ctx)
    for fn in (fig2, fig3, fig4, fig5, fig6, supp, fig_forest, fig_clean_vs_loss, fig_casemix, fig_sensitivity,
               fig_examples):
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001
            log.warning("figure %s skipped: %s", fn.__name__, e)
            plt.close("all")
    log.info("figures -> %s", ctx.out / "figures")
