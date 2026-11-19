"""Fast patient-weighted performance engine and two-level (backbone x patient) bootstrap.

A *cell* is a set of prediction rows of one (configuration, seed), e.g. 'clean' (HQ test images),
'real_any' (all LQ test images) or a corruption condition. Each cell is pre-sorted once, so a
bootstrap replicate only needs weighted counts: patient resampling is expressed as integer weights
(multiplicities) per patient, which gives exactly the same AUROC as physically duplicating rows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

log = logging.getLogger(__name__)


@dataclass
class Cell:
    y: np.ndarray          # binary target, sorted by score
    gid: np.ndarray        # tie-group id per row (sorted)
    pc: np.ndarray         # patient code per row (sorted)
    pred: np.ndarray       # prediction at frozen threshold (sorted)
    ng: int
    ovr: list = field(default_factory=list)  # multi-class: one-vs-rest sub-cells (y, gid, pc, ng)


def _prep(y: np.ndarray, s: np.ndarray, pc: np.ndarray, pred: np.ndarray) -> Cell:
    o = np.argsort(s, kind="mergesort")
    s_sorted = s[o]
    gid = np.concatenate([[0], np.cumsum(np.diff(s_sorted) != 0)]).astype(np.int64)
    return Cell(y[o].astype(np.float32), gid.astype(np.int32), pc[o].astype(np.int32), pred[o].astype(np.float32),
                int(gid[-1]) + 1 if len(gid) else 0)


def wauc(y: np.ndarray, gid: np.ndarray, ng: int, w: np.ndarray) -> float:
    pos = np.bincount(gid, weights=w * y, minlength=ng)
    neg = np.bincount(gid, weights=w * (1 - y), minlength=ng)
    wp, wn = pos.sum(), neg.sum()
    if wp <= 0 or wn <= 0:
        return np.nan
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (wp * wn))


def cell_metric(c: Cell, wpat: np.ndarray, metric: str) -> float:
    w = wpat[c.pc]
    if metric == "auroc":
        if c.ovr:
            vals = [wauc(yk, gk, ngk, wpat[pk]) for yk, gk, pk, ngk in c.ovr]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else np.nan
        return wauc(c.y, c.gid, c.ng, w)
    if metric == "sens":
        d = (w * c.y).sum()
        return float((w * c.y * c.pred).sum() / d) if d > 0 else np.nan
    if metric == "spec":
        d = (w * (1 - c.y)).sum()
        return float((w * (1 - c.y) * (1 - c.pred)).sum() / d) if d > 0 else np.nan
    if metric == "acc":
        d = w.sum()
        correct = c.pred == c.y
        return float((w * correct).sum() / d) if d > 0 else np.nan
    raise ValueError(metric)


class Engine:
    """Performance of every (configuration, seed, cell) under arbitrary patient weights."""

    def __init__(self, df: pd.DataFrame, cells: dict[str, pd.Series], metric: str = "auroc", n_classes: int = 2,
                 negative_class: int = 0):
        df = df[df["label"].notna()].copy()
        df["config"] = df["backbone"].astype(str) + "/" + df["mode"].astype(str)
        self.metric, self.n_classes = metric, n_classes
        self.configs = sorted(df["config"].unique().tolist())
        self.seeds = sorted(df["seed"].unique().tolist())
        self.cells = list(cells.keys())
        self.backbones = sorted({c.split("/")[0] for c in self.configs})
        self.config_backbone = np.array([self.backbones.index(c.split("/")[0]) for c in self.configs])
        codes, uniq = pd.factorize(df["patient_id"].astype(str))
        self.patients = np.asarray(uniq)
        self.n_pat = len(uniq)
        df["_pc"] = codes
        lab = df["label"].astype(int).to_numpy()
        ybin = lab if n_classes == 2 else (lab != negative_class).astype(int)
        df["_y"] = ybin
        self.grid: dict[tuple[int, int, int], Cell] = {}
        masks = {k: v.reindex(df.index).fillna(False).to_numpy(bool) for k, v in cells.items()}
        cidx = {c: i for i, c in enumerate(self.configs)}
        sidx = {s: i for i, s in enumerate(self.seeds)}
        for (cfg, seed), g in df.groupby(["config", "seed"]):
            pos = df.index.get_indexer(g.index)
            for k, name in enumerate(self.cells):
                m = masks[name][pos]
                if not m.any():
                    continue
                sub = g[m]
                y = sub["_y"].to_numpy()
                pc = sub["_pc"].to_numpy()
                pred = sub["pred_at_threshold"].to_numpy()
                if n_classes == 2 or metric != "auroc":
                    c = _prep(y, sub["prob"].to_numpy(np.float64), pc, pred)
                else:
                    c = _prep(y, sub["prob"].to_numpy(np.float64), pc, pred)
                    labs = sub["label"].astype(int).to_numpy()
                    for kk in range(n_classes):
                        sc = sub[f"prob_{kk}"].to_numpy(np.float64)
                        yk = (labs == kk).astype(int)
                        o = np.argsort(sc, kind="mergesort")
                        gid = np.concatenate([[0], np.cumsum(np.diff(sc[o]) != 0)]).astype(np.int64)
                        c.ovr.append((yk[o].astype(np.float32), gid.astype(np.int32), pc[o].astype(np.int32), int(gid[-1]) + 1))
                self.grid[(cidx[cfg], sidx[seed], k)] = c

    def compute(self, wpat: np.ndarray | None = None, metric: str | None = None) -> np.ndarray:
        """Array (n_configs, n_seeds, n_cells); NaN where a cell is absent or undefined."""
        metric = metric or self.metric
        w = np.ones(self.n_pat) if wpat is None else wpat
        out = np.full((len(self.configs), len(self.seeds), len(self.cells)), np.nan)
        for (i, s, k), c in self.grid.items():
            out[i, s, k] = cell_metric(c, w, metric)
        return out

    def seed_mean(self, arr: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(arr, axis=1)


def _one_replicate(engine: Engine, seed: int, inner: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nb = len(engine.backbones)
    bb = rng.integers(0, nb, size=nb)
    acc = None
    for _ in range(max(1, inner)):
        pidx = rng.integers(0, engine.n_pat, size=engine.n_pat)
        w = np.bincount(pidx, minlength=engine.n_pat).astype(np.float64)
        v = engine.seed_mean(engine.compute(w, metric))
        acc = v if acc is None else acc + v
    return acc / max(1, inner), bb


def _chunk(engine: Engine, seeds: list[int], inner: int, metric: str) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_one_replicate(engine, s, inner, metric) for s in seeds]


def two_level_bootstrap(engine: Engine, n_boot: int, seed: int = 0, inner: int = 1, n_jobs: int = -1,
                        metric: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (perf (n_boot, n_configs, n_cells) seed-averaged, backbone samples (n_boot, n_backbones)).

    Replicates are split into one chunk per worker so the engine is serialised only once per worker.
    """
    from joblib import effective_n_jobs

    metric = metric or engine.metric
    seeds = [seed * 100003 + b for b in range(n_boot)]
    nw = max(1, min(effective_n_jobs(n_jobs), n_boot))
    chunks = [seeds[i::nw] for i in range(nw)]
    out = Parallel(n_jobs=nw)(delayed(_chunk)(engine, c, inner, metric) for c in chunks)
    res: list = [None] * n_boot
    for i, c in enumerate(out):
        for j, r in enumerate(c):
            res[i + j * nw] = r
    perf = np.stack([r[0] for r in res])
    bbs = np.stack([r[1] for r in res])
    return perf, bbs


def config_index_for_backbones(engine: Engine, bb_sample: np.ndarray) -> np.ndarray:
    """Indices of all configurations of the sampled backbones (with multiplicity)."""
    idx = [np.flatnonzero(engine.config_backbone == b) for b in bb_sample]
    return np.concatenate(idx) if idx else np.array([], int)


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))


def patient_bootstrap(fn, df: pd.DataFrame, n_boot: int, seed: int = 0, group: str = "patient_id", n_jobs: int = 1) -> np.ndarray:
    """Generic patient-cluster bootstrap of a statistic ``fn(df) -> float``."""
    groups = df[group].astype(str).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = [np.flatnonzero(inv == i) for i in range(len(uniq))]

    def one(b):
        rng = np.random.default_rng(seed * 7919 + b)
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        return fn(df.iloc[idx])

    return np.asarray(Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in range(n_boot)), dtype=float)
