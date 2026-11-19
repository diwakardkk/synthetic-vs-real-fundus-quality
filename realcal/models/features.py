"""Frozen-feature extraction and caching (linear probes, Level-2 encoder, quality detector).

Cache: ``cache/features/<dataset>/<encoder>/<size>/<condition>.npy`` (float16, N x D) with the image
order in ``<condition>_ids.parquet``. Features are fold-independent, so they are computed once.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from realcal.config import Ctx
from realcal.data.datasets import FundusDataset, make_loader
from realcal.jobs import Job
from realcal.models.backbones import Backbone, build_backbone
from realcal.utils.io import save_npy, write_parquet
from realcal.utils.signals import check_stop

log = logging.getLogger(__name__)


def feat_paths(ctx: Ctx, dataset: str, encoder: str, size: int, condition: str) -> tuple[Path, Path]:
    d = ctx.out / "cache" / "features" / dataset / encoder / str(size)
    return d / f"{condition}.npy", d / f"{condition}_ids.parquet"


def image_ids_for(ctx: Ctx, dataset: str, image_set: str) -> list[str]:
    from realcal.corruptions.cache import eval_hq_ids
    from realcal.data.common import load_metadata

    if image_set == "hq_eval":
        return eval_hq_ids(ctx, dataset)
    meta = load_metadata(ctx, dataset)
    if image_set == "all":
        return meta["image_id"].tolist()
    if image_set == "ddr_entropy":
        return ddr_entropy_ids(ctx)
    raise ValueError(image_set)


def ddr_entropy_ids(ctx: Ctx) -> list[str]:
    from realcal.data.common import load_metadata

    meta = load_metadata(ctx, "ddr")
    ung = meta.loc[meta["ungradable"], "image_id"].tolist()
    grad = meta.loc[~meta["ungradable"], "image_id"]
    n = int(ctx.cfg.external.ddr_max_gradable_eval)
    grad = grad.sample(n=min(n, len(grad)), random_state=0).tolist()
    return sorted(ung + grad)


@torch.no_grad()
def extract(ctx: Ctx, bb: Backbone, dataset: str, size: int, condition: str, ids: list[str], batch_size: int,
            device: torch.device) -> np.ndarray:
    ds = FundusDataset(ctx, dataset, ids, None, size, bb.mean, bb.std, condition=condition, train=False)
    nw = int(ctx.extra.get("num_workers", 4))
    loader = make_loader(ds, batch_size, shuffle=False, num_workers=nw)
    out = np.zeros((len(ids), bb.embed_dim), dtype=np.float16)
    use_amp = device.type == "cuda"
    for x, _, idx in loader:
        check_stop()
        x = x.to(device, non_blocking=True)
        if bb.is_cnn:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            f = bb(x)
        out[idx.numpy()] = f.float().cpu().numpy().astype(np.float16)
    return out


def get_features(ctx: Ctx, encoder: str, dataset: str, size: int, condition: str, ids: list[str],
                 bb: Backbone | None = None, batch_size: int = 128) -> np.ndarray:
    """Load cached features for ``ids`` (computing and caching any that are missing)."""
    import fcntl

    fp, ip = feat_paths(ctx, dataset, encoder, size, condition)
    fp.parent.mkdir(parents=True, exist_ok=True)

    def _read():
        if fp.exists() and ip.exists():
            return pd.read_parquet(ip)["image_id"].tolist(), np.load(fp, mmap_mode="r")
        return [], None

    cached_ids, feats = _read()
    pos = {k: i for i, k in enumerate(cached_ids)}
    if any(i not in pos for i in ids):
        lock = open(fp.with_suffix(".lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)  # another worker may be writing the same cache file
        try:
            cached_ids, feats = _read()
            pos = {k: i for i, k in enumerate(cached_ids)}
            feats, pos = _fill_missing(ctx, encoder, dataset, size, condition, ids, bb, batch_size, cached_ids, feats, pos, fp, ip)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    return np.asarray(feats[[pos[i] for i in ids]], dtype=np.float32)


def _fill_missing(ctx, encoder, dataset, size, condition, ids, bb, batch_size, cached_ids, feats, pos, fp, ip):
    missing = [i for i in ids if i not in pos]
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if bb is None:
            bb = build_backbone(ctx, encoder, size).to(device).eval()
        new = extract(ctx, bb, dataset, size, condition, missing, batch_size, device)
        all_ids = cached_ids + missing
        allf = np.concatenate([np.asarray(feats) if feats is not None else np.zeros((0, new.shape[1]), np.float16), new])
        save_npy(fp, allf)
        write_parquet(ip, pd.DataFrame({"image_id": all_ids}))
        feats, pos = allf, {k: i for i, k in enumerate(all_ids)}
    return feats, pos


def run_features_job(job: Job, ctx: Ctx) -> None:
    p = job.params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = build_backbone(ctx, p["encoder"], int(p["size"])).to(device).eval()
    if bb.is_cnn:
        bb = bb.to(memory_format=torch.channels_last)
    for cond in p["conditions"]:
        image_set = p["image_set"] if cond == "clean" else "hq_eval"
        ids = image_ids_for(ctx, p["dataset"], image_set)
        fp, ip = feat_paths(ctx, p["dataset"], p["encoder"], int(p["size"]), cond)
        if fp.exists() and ip.exists() and set(pd.read_parquet(ip)["image_id"]) >= set(ids):
            continue
        get_features(ctx, p["encoder"], p["dataset"], int(p["size"]), cond, ids, bb=bb, batch_size=int(p["batch_size"]))
        log.info("features %s %s %s: %d images", p["encoder"], p["dataset"], cond, len(ids))
