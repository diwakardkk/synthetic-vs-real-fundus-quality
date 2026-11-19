"""Read-only access to the prediction store. All analyses read predictions only through here."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from realcal.config import Ctx

log = logging.getLogger(__name__)


def load_predictions(ctx: Ctx, eval_dataset: str, train_dataset: str | None = None, variant: str = "main",
                     conditions: list[str] | None = None, backbones: list[str] | None = None,
                     columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate prediction Parquet files matching the filters (empty frame if none)."""
    root = ctx.out / "predictions" / eval_dataset
    if not root.exists():
        return pd.DataFrame()
    train_dataset = train_dataset or eval_dataset
    prefix = f"train__{train_dataset}__" if variant == "main" else f"train__{train_dataset}-{variant}__"
    want = set(conditions) if conditions is not None else None
    frames = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if backbones is not None and d.name.split("__")[2] not in backbones:
            continue
        for f in d.glob("*.parquet"):
            if want is not None and f.stem not in want:
                continue
            frames.append(pd.read_parquet(f, columns=columns))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


def config_key(df: pd.DataFrame) -> pd.Series:
    return df["backbone"].astype(str) + "/" + df["mode"].astype(str)
