"""Robustness checks added after the first full DeepDRiD run.

* :func:`noise_ceiling` - how reproducible is the ranking of configurations by real loss?
  - patient split-half reliability (test-sample noise): split patients at random into halves, compute
    each configuration's real loss on each half, correlate the halves (Spearman), Spearman-Brown to
    full size; repeated ``n_splits`` times (median).
  - seed reliability (training noise): mean pairwise Spearman between seeds, Spearman-Brown to the
    mean of k seeds.
  - combined reliability ~ rel_patient + rel_seed - 1 (independent noise sources), and the ceiling
    sqrt(reliability): the largest Spearman any predictor can be expected to reach against the
    observed real-loss ranking.
* :func:`standardized_real_loss` - real loss with LQ images re-weighted so that the distribution of a
  case-mix variable (DR grade for DeepDRiD, disease class for FIVES) matches the HQ images. Removes
  the confounding of quality with disease severity / class seen in the dataset tables.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from realcal.config import Ctx
from realcal.eval.store import load_predictions
from realcal.stats.bootstrap import Engine
from realcal.stats.tests import spearman

log = logging.getLogger(__name__)


def _sb(r: float, k: float) -> float:
    """Spearman-Brown prophecy for k-fold lengthening."""
    if r is None or np.isnan(r):
        return np.nan
    return float(k * r / (1 + (k - 1) * r)) if (1 + (k - 1) * r) != 0 else np.nan


def noise_ceiling(ctx: Ctx, dataset: str = "deepdrid", train_ds: str | None = None, variant: str = "main",
                  stratum: str = "any", n_splits: int = 200, seed: int = 0) -> dict:
    train_ds = train_ds or dataset
    df = load_predictions(ctx, dataset, train_ds, variant, conditions=["clean", "real"])
    if df.empty:
        return {"error": "no predictions"}
    df = df[df["label"].notna()]
    cond = df["condition"].astype(str)
    col = "lq_any" if stratum == "any" else f"lq_{stratum}"
    cells = {"clean": cond == "clean", "real": (cond == "real") & df[col].astype(bool)}
    n_classes = int(ctx.ds_cfg(train_ds).n_classes)
    eng = Engine(df, cells, metric="auroc", n_classes=n_classes,
                 negative_class=int(ctx.cfg.threshold.multiclass_negative_class))
    full = eng.compute()                                   # (configs, seeds, 2)
    d_seed = full[:, :, 0] - full[:, :, 1]                  # loss per config x seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y = np.nanmean(d_seed, axis=1)
        sd_seed = np.nanstd(d_seed, axis=1, ddof=1) if d_seed.shape[1] > 1 else np.full(len(y), np.nan)

    # seed reliability
    k = d_seed.shape[1]
    pair_r = [spearman(d_seed[:, a], d_seed[:, b]) for a in range(k) for b in range(a + 1, k)]
    r_seed = float(np.nanmean(pair_r)) if pair_r else np.nan
    rel_seed = _sb(r_seed, k) if k > 1 else np.nan

    # patient split-half reliability
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(int(n_splits)):
        m = rng.random(eng.n_pat) < 0.5
        a = eng.compute(m.astype(float))
        b = eng.compute((~m).astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            da = np.nanmean(a[:, :, 0] - a[:, :, 1], axis=1)
            db = np.nanmean(b[:, :, 0] - b[:, :, 1], axis=1)
        rs.append(spearman(da, db))
    rs = np.asarray(rs, float)
    r_half = float(np.nanmedian(rs)) if np.isfinite(rs).any() else np.nan
    rel_pat = _sb(r_half, 2)

    rel_comb = np.nan
    if np.isfinite(rel_pat) and np.isfinite(rel_seed):
        rel_comb = float(max(0.0, rel_pat + rel_seed - 1))
    elif np.isfinite(rel_pat):
        rel_comb = float(max(0.0, rel_pat))
    ceil = lambda r: float(np.sqrt(max(r, 0.0))) if np.isfinite(r) else np.nan  # noqa: E731
    out = {
        "dataset": dataset, "train_dataset": train_ds, "stratum": stratum, "n_configs": int(len(y)),
        "n_seeds": int(k), "n_patients": int(eng.n_pat), "n_splits": int(n_splits),
        "loss_sd_between_configs": float(np.nanstd(y, ddof=1)),
        "loss_range": [float(np.nanmin(y)), float(np.nanmax(y))],
        "median_seed_sd": float(np.nanmedian(sd_seed)),
        "seed_pairwise_spearman": r_seed, "reliability_seed": rel_seed,
        "split_half_spearman_median": r_half,
        "split_half_spearman_iqr": [float(np.nanquantile(rs, 0.25)), float(np.nanquantile(rs, 0.75))],
        "reliability_patient": rel_pat,
        "reliability_combined": rel_comb,
        "ceiling_rho_patient": ceil(rel_pat), "ceiling_rho_seed": ceil(rel_seed),
        "ceiling_rho_combined": ceil(rel_comb),
    }
    log.info("noise ceiling %s: rel_patient %.2f, rel_seed %.2f -> max attainable rho ~ %.2f",
             dataset, out["reliability_patient"], out["reliability_seed"], out["ceiling_rho_combined"])
    return out


# ----------------------------------------------------------------------------- case-mix standardisation

def _auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s, sample_weight=w))


def _macro_ovr(lab: np.ndarray, P: np.ndarray, w: np.ndarray | None = None) -> float:
    vals = [_auc((lab == k).astype(int), P[:, k], w) for k in range(P.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def _weights(strat_lq: pd.Series, strat_hq: pd.Series) -> np.ndarray:
    """w(s) = P_HQ(s) / P_LQ(s) for each LQ row; strata absent from HQ get weight 0."""
    p_hq = strat_hq.value_counts(normalize=True)
    p_lq = strat_lq.value_counts(normalize=True)
    ratio = (p_hq / p_lq).reindex(p_lq.index).fillna(0.0)
    return strat_lq.map(ratio).fillna(0.0).to_numpy(float)


def standardized_real_loss(ctx: Ctx, dataset: str, by: str, train_ds: str | None = None,
                           variant: str = "main", stratum: str = "any") -> pd.DataFrame:
    """Per configuration: raw and case-mix-standardised real loss (HQ AUROC - LQ AUROC).

    ``by`` is a metadata column (``grade`` for DeepDRiD, ``label`` for FIVES). The LQ stratum is
    re-weighted to the HQ distribution of ``by``; the HQ AUROC is unchanged.
    """
    from realcal.data.common import load_metadata

    train_ds = train_ds or dataset
    df = load_predictions(ctx, dataset, train_ds, variant, conditions=["clean", "real"])
    if df.empty:
        return pd.DataFrame()
    df = df[df["label"].notna()].copy()
    meta = load_metadata(ctx, dataset)[["image_id", by]].rename(columns={by: "_strat"})
    df = df.drop(columns=[c for c in ("_strat",) if c in df]).merge(meta, on="image_id", how="left")
    df["_strat"] = df["_strat"].astype("string").fillna("NA")
    n_classes = int(ctx.ds_cfg(train_ds).n_classes)
    pcols = [f"prob_{k}" for k in range(n_classes)] if n_classes > 2 else None
    rows = []
    for (b, m, s), g in df.groupby(["backbone", "mode", "seed"]):
        cond = g["condition"].astype(str)
        hq = g[cond == "clean"]
        lq = g[(cond == "real") & g["lq_any" if stratum == "any" else f"lq_{stratum}"].astype(bool)]
        if not len(hq) or not len(lq):
            continue
        w = _weights(lq["_strat"], hq["_strat"])
        if n_classes > 2:
            a_hq = _macro_ovr(hq["label"].astype(int).to_numpy(), hq[pcols].to_numpy(float))
            a_lq = _macro_ovr(lq["label"].astype(int).to_numpy(), lq[pcols].to_numpy(float))
            a_lq_w = _macro_ovr(lq["label"].astype(int).to_numpy(), lq[pcols].to_numpy(float), w)
        else:
            a_hq = _auc(hq["label"].astype(int).to_numpy(), hq["prob"].to_numpy(float))
            a_lq = _auc(lq["label"].astype(int).to_numpy(), lq["prob"].to_numpy(float))
            a_lq_w = _auc(lq["label"].astype(int).to_numpy(), lq["prob"].to_numpy(float), w)
        rows.append({"config": f"{b}/{m}", "seed": s, "auroc_hq": a_hq, "auroc_lq": a_lq, "auroc_lq_std": a_lq_w,
                     "eff_n_lq": float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.groupby("config").mean(numeric_only=True).reset_index().drop(columns=["seed"])
    t["delta_raw"] = t["auroc_hq"] - t["auroc_lq"]
    t["delta_std"] = t["auroc_hq"] - t["auroc_lq_std"]
    t["standardized_by"] = by
    t["stratum"] = stratum
    return t


def summarize_standardized(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {}
    return {"by": str(t["standardized_by"].iloc[0]), "stratum": str(t["stratum"].iloc[0]), "n_configs": int(len(t)),
            "delta_raw_median": float(t["delta_raw"].median()), "delta_std_median": float(t["delta_std"].median()),
            "delta_std_range": [float(t["delta_std"].min()), float(t["delta_std"].max())],
            "rank_agreement_raw_vs_std": spearman(t["delta_raw"].to_numpy(), t["delta_std"].to_numpy()),
            "effective_n_lq_median": float(t["eff_n_lq"].median())}
