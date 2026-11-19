"""Download all pretrained weights once into ``weights_root`` (run with internet access).

Hugging Face entries (``hf_repo`` [+ ``hf_file``]) use ``huggingface_hub`` and the user's token
(``huggingface-cli login`` or ``HF_TOKEN``) for gated models (RETFound, DINOv3: request access on
the model page first). Entries with a ``url`` are fetched with ``gdown`` (Google Drive) or urllib.
Missing weights are reported; training skips those backbones without stopping other jobs.
"""
from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

from realcal.config import Ctx
from realcal.models.backbones import weights_available, weights_path

log = logging.getLogger(__name__)


def _fetch_one(ctx: Ctx, key: str, c) -> tuple[str, str]:
    dst = weights_path(ctx, key)
    if weights_available(ctx, key):
        return key, f"present  {dst}"
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        if c.get("hf_repo"):
            from huggingface_hub import hf_hub_download, snapshot_download

            token = os.environ.get("HF_TOKEN")
            if c.get("hf_file"):
                p = hf_hub_download(str(c.hf_repo), str(c.hf_file), local_dir=str(dst.parent), token=token)
                if Path(p) != dst and Path(p).exists() and not dst.exists():
                    os.replace(p, dst)
            else:
                snapshot_download(str(c.hf_repo), local_dir=str(dst), token=token)
            return key, f"downloaded  {dst}"
        url = str(c.get("url", "") or "")
        if url:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if "drive.google.com" in url:
                import gdown

                gdown.download(url, str(dst), quiet=False, fuzzy=True)
            else:
                urllib.request.urlretrieve(url, dst)  # noqa: S310
            return key, f"downloaded  {dst}"
        return key, f"MANUAL DOWNLOAD NEEDED -> place the file at {dst} (see README, 'Weights')"
    except Exception as e:  # noqa: BLE001
        return key, f"FAILED ({type(e).__name__}: {str(e)[:200]}) -> place the file at {dst}"


def fetch_all(ctx: Ctx) -> dict[str, str]:
    ctx.weights_root.mkdir(parents=True, exist_ok=True)
    entries = dict(ctx.cfg.backbones)
    entries.update(dict(ctx.cfg.analysis_encoders))
    res = dict(_fetch_one(ctx, k, c) for k, c in entries.items())
    print("\nWEIGHTS STATUS")
    for k, s in res.items():
        print(f"  {k:14s} {s}")
    return res
