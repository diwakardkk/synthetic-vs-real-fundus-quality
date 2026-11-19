"""Shared analysis object: pooled out-of-fold performance per (configuration, seed, cell), seed means,
two-level bootstrap replicates, real and synthetic losses, and rank-agreement statistics with CIs.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.corruptions.registry import active_b2, active_main, groups
from realcal.eval.store import load_predictions
from realcal.stats.bootstrap import Engine, config_index_for_backbones, percentile_ci, two_level_bootstrap
from realcal.stats.tests import RANK_METRICS, topk_overlap

log = logging.getLogger(__name__)

REAL_STRATA = ["any", "artifact", "clarity", "field"]


@dataclass
class Analysis:
    key: str
    metric: str
    engine: Engine
    configs: list[str]
    cells: list[str]
    perf_seed: np.ndarray            # (n_configs, n_seeds, n_cells)
    perf: np.ndarray                 # (n_configs, n_cells) seed mean
    perf_sd: np.ndarray              # (n_configs, n_cells) seed SD
    boot: np.ndarray | None          # (n_boot, n_configs, n_cells)
    boot_bb: np.ndarray | None       # (n_boot, n_backbones)
    syn_conditions: list[str]
    b2_conditions: list[str]
    fingerprint: str = ""

    def ci(self, name: str) -> int:
        return self.cells.index(name)

    def col(self, name: str, boot: bool = False) -> np.ndarray:
        k = self.ci(name)
        return self.boot[:, :, k] if boot else self.perf[:, k]

    @property
    def backbones(self) -> list[str]:
        return self.engine.backbones

    def family_of(self, ctx: Ctx) -> list[str]:
        return [ctx.family(c.split("/")[0]) for c in self.configs]


def store_fingerprint(ctx: Ctx, eval_ds: str, train_ds: str, variant: str) -> str:
    """Number, total size and latest mtime of the matching prediction files (cache key)."""
    import os

    root = ctx.out / "predictions" / eval_ds
    prefix = f"train__{train_ds}__" if variant == "main" else f"train__{train_ds}-{variant}__"
    n, size, mt = 0, 0, 0.0
    if root.exists():
        for d in os.scandir(root):
            if d.is_dir() and d.name.startswith(prefix):
                for f in os.scandir(d.path):
                    if f.name.endswith(".parquet"):
                        st = f.stat()
                        n, size, mt = n + 1, size + st.st_size, max(mt, st.st_mtime)
    return f"{n}|{size}|{mt:.0f}"


def build_cells(df: pd.DataFrame, syn: list[str]) -> dict[str, pd.Series]:
    cond = df["condition"].astype(str)
    cells = {"clean": cond == "clean"}
    for q in REAL_STRATA:
        col = "lq_any" if q == "any" else f"lq_{q}"
        cells[f"real_{q}"] = (cond == "real") & df[col].astype(bool)
    for s in syn:
        cells[s] = cond == s
    return cells


def build_analysis(ctx: Ctx, eval_ds: str, train_ds: str | None = None, variant: str = "main",
                   metric: str = "auroc", with_synth: bool = True, with_boot: bool = True,
                   relative: bool = False, cache: bool = True) -> Analysis | None:
    """Assemble (and cache) the analysis object for one evaluation setting."""
    train_ds = train_ds or eval_ds
    key = f"{eval_ds}_from_{train_ds}_{variant}_{metric}_{'syn' if with_synth else 'real'}"
    cpath = ctx.p("cache", "analysis", f"{key}.pkl")
    fp = store_fingerprint(ctx, eval_ds, train_ds, variant) + f"|boot={with_boot}|{ctx.cfg.stats.n_boot}|{ctx.cfg.stats.n_boot_inner}"
    if cache and cpath.exists():
        with open(cpath, "rb") as f:
            cached = pickle.load(f)
        if getattr(cached, "fingerprint", None) == fp:
            return cached
    syn = [c.name for c in active_main(ctx)] if with_synth else []
    b2 = [c.name for c in active_b2(ctx)] if with_synth else []
    conds = ["clean", "real"] + syn + b2
    df = load_predictions(ctx, eval_ds, train_ds, variant, conditions=conds)
    if df.empty:
        log.warning("no predictions for %s", key)
        return None
    n_classes = int(ctx.ds_cfg(train_ds).n_classes)
    cells = build_cells(df, syn + b2)
    eng = Engine(df, cells, metric=metric, n_classes=n_classes,
                 negative_class=int(ctx.cfg.threshold.multiclass_negative_class))
    perf_seed = eng.compute()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        perf = np.nanmean(perf_seed, axis=1)
        perf_sd = np.nanstd(perf_seed, axis=1, ddof=1) if perf_seed.shape[1] > 1 else np.zeros_like(perf)
    boot = bbs = None
    if with_boot:
        sc = ctx.cfg.stats
        boot, bbs = two_level_bootstrap(eng, int(sc.n_boot), seed=17, inner=int(sc.n_boot_inner), n_jobs=int(sc.n_jobs))
    an = Analysis(key, metric, eng, eng.configs, eng.cells, perf_seed, perf, perf_sd, boot, bbs, syn, b2)
    an.fingerprint = fp
    if cache:
        with open(cpath, "wb") as f:
            pickle.dump(an, f)
    log.info("analysis %s: %d configs x %d seeds x %d cells", key, len(eng.configs), len(eng.seeds), len(eng.cells))
    return an


# ----------------------------------------------------------------------------- losses

def _loss(clean: np.ndarray, other: np.ndarray, metric: str, relative: bool) -> np.ndarray:
    d = clean - other
    if relative and metric == "auroc":
        with np.errstate(all="ignore"):
            d = d / (clean - 0.5)
    return d


def delta_real(an: Analysis, q: str = "any", boot: bool = False, relative: bool = False) -> np.ndarray:
    return _loss(an.col("clean", boot), an.col(f"real_{q}", boot), an.metric, relative)


def delta_syn(an: Analysis, boot: bool = False, conditions: list[str] | None = None, relative: bool = False) -> np.ndarray:
    """(n_configs, n_conditions) or (n_boot, n_configs, n_conditions)."""
    conditions = conditions or an.syn_conditions
    ks = [an.ci(c) for c in conditions]
    if boot:
        clean = an.boot[:, :, an.ci("clean")][..., None]
        return _loss(clean, an.boot[:, :, ks], an.metric, relative)
    clean = an.perf[:, an.ci("clean")][:, None]
    return _loss(clean, an.perf[:, ks], an.metric, relative)


def group_scores(ctx: Ctx, an: Analysis, ds: np.ndarray, conditions: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Mean synthetic loss per corruption group (last axis = groups). Works for 2-D or 3-D ``ds``."""
    from realcal.corruptions.registry import by_name

    conditions = conditions or an.syn_conditions
    lookup = by_name(ctx)
    gs = [g for g in groups(ctx) if any(lookup[c].group == g for c in conditions)]
    cols = []
    import warnings

    for g in gs:
        idx = [i for i, c in enumerate(conditions) if lookup[c].group == g]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cols.append(np.nanmean(ds[..., idx], axis=-1))
    return np.stack(cols, axis=-1), gs


# ----------------------------------------------------------------------------- rank statistics

def rank_stats(an: Analysis, pred: np.ndarray, real: np.ndarray, pred_boot: np.ndarray | None,
               real_boot: np.ndarray | None, level: float = 0.95) -> dict:
    """Point estimates and two-level bootstrap CIs of spearman/kendall/pra/mae (+ top-5 overlap)."""
    out: dict = {}
    for name, fn in RANK_METRICS.items():
        out[name] = fn(pred, real)
    out["top5"] = topk_overlap(pred, real, 5)
    if pred_boot is not None and real_boot is not None and an.boot_bb is not None:
        reps = {k: [] for k in RANK_METRICS}
        for b in range(pred_boot.shape[0]):
            idx = config_index_for_backbones(an.engine, an.boot_bb[b])
            for name, fn in RANK_METRICS.items():
                reps[name].append(fn(pred_boot[b, idx], real_boot[b, idx]))
        for name in RANK_METRICS:
            arr = np.asarray(reps[name], float)
            out[f"{name}_ci"] = list(percentile_ci(arr, level))
            out[f"{name}_boot"] = arr
    return out


def strip_boot(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.endswith("_boot")}


def config_frame(ctx: Ctx, an: Analysis) -> pd.DataFrame:
    rows = []
    for c in an.configs:
        b, m = c.split("/")
        rows.append({"config": c, "backbone": b, "mode": m, "family": ctx.family(b)})
    return pd.DataFrame(rows)
