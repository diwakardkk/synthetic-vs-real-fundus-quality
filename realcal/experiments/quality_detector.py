"""H3: quality detectors trained on synthetic corruptions of DeepDRiD HQ images.

Frozen encoder (config ``features.qdet_encoder``, default DINOv2 ViT-B) + logistic-regression head
separating clean DeepDRiD HQ images (0) from corrupted DeepDRiD HQ images (1). One detector uses the
uniform mix over the 75 conditions, one the calibrated mix (frozen Level-2 pi). Tested with no
adaptation on MSHF (overall-quality labels) and DDR (gradable vs ungradable).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from realcal.config import Ctx
from realcal.corruptions.cache import eval_hq_ids
from realcal.models.features import get_features

log = logging.getLogger(__name__)


def train_detector(ctx: Ctx, mix: dict[str, float], seed: int = 0):
    enc = str(ctx.cfg.features.qdet_encoder)
    ids = eval_hq_ids(ctx, "deepdrid")
    clean = get_features(ctx, enc, "deepdrid", 224, "clean", ids)
    names = [k for k, v in mix.items() if v > 0]
    p = np.asarray([mix[k] for k in names], float)
    p /= p.sum()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(names), size=len(ids), p=p)
    synth = np.zeros_like(clean)
    for k in np.unique(pick):
        rows = np.flatnonzero(pick == k)
        f = get_features(ctx, enc, "deepdrid", 224, names[k], [ids[i] for i in rows])
        synth[rows] = f
    X = np.concatenate([clean, synth])
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(synth))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    clf.fit(X, y)
    return clf


def score_dataset(ctx: Ctx, clf, dataset: str, ids: list[str]) -> np.ndarray:
    enc = str(ctx.cfg.features.qdet_encoder)
    X = get_features(ctx, enc, dataset, 224, "clean", ids)
    return clf.predict_proba(X)[:, 1]


def compare_detectors(y: np.ndarray, s_uni: np.ndarray, s_cal: np.ndarray, level: float = 0.95) -> dict:
    from realcal.stats.tests import delong_ci, delong_test

    a_u = delong_ci(y, s_uni, level)
    a_c = delong_ci(y, s_cal, level)
    t = delong_test(y, s_cal, s_uni)
    return {"n": int(len(y)), "n_pos": int(np.sum(y)), "auroc_uniform": a_u[0], "auroc_uniform_ci": list(a_u[1:]),
            "auroc_realcal": a_c[0], "auroc_realcal_ci": list(a_c[1:]), "delong_diff": t["diff"], "delong_p": t["p"]}


def detectors_from_frozen(ctx: Ctx, frozen: dict):
    conds = list(frozen["level2"]["conditions"])
    uni = {c: 1.0 / len(conds) for c in conds}
    cal = dict(zip(conds, frozen["level2"]["pi"]))
    return train_detector(ctx, uni, 0), train_detector(ctx, cal, 0)


def evaluate(ctx: Ctx, frozen: dict, dataset: str, meta: pd.DataFrame, target: str, by: str | None = None) -> tuple[dict, pd.DataFrame]:
    """AUROC of both detectors for ``target`` (bool column) on ``dataset``, overall and by ``by``."""
    det_u, det_c = detectors_from_frozen(ctx, frozen)
    m = meta[meta[target].notna()].copy()
    if dataset == "mshf":
        m = m[m["q_overall"].notna()]
    ids = m["image_id"].tolist()
    m["score_uniform"] = score_dataset(ctx, det_u, dataset, ids)
    m["score_realcal"] = score_dataset(ctx, det_c, dataset, ids)
    y = m[target].astype(int).to_numpy()
    lvl = float(ctx.cfg.stats.ci)
    res = {"overall": compare_detectors(y, m["score_uniform"].to_numpy(), m["score_realcal"].to_numpy(), lvl)}
    if by:
        for k, g in m.groupby(by):
            yy = g[target].astype(int).to_numpy()
            if 0 < yy.sum() < len(yy):
                res[f"{by}={k}"] = compare_detectors(yy, g["score_uniform"].to_numpy(), g["score_realcal"].to_numpy(), lvl)
    return res, m[["image_id", target] + ([by] if by else []) + ["score_uniform", "score_realcal"]]
