"""Atomic file writes and small I/O helpers.

Every write goes to ``<name>.tmp.<pid>`` first and is then moved into place with
``os.replace`` so that a crash never leaves a half-written file behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _tmp(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if pd.isna(o) if not isinstance(o, (list, dict, tuple)) else False:
        return None
    return str(o)


def write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_csv(path: str | Path, df: pd.DataFrame, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def torch_save(path: str | Path, obj: Any) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_png(path: str | Path, img: np.ndarray) -> None:
    """Save a uint8 HxW or HxWx3 (RGB) array as lossless PNG, atomically."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = img if img.ndim == 2 else img[..., ::-1]
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"PNG encode failed for {path}")
    atomic_write_bytes(path, buf.tobytes())


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image as uint8 RGB."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image

        img = np.asarray(Image.open(path).convert("RGB"))
        return img
    return img[..., ::-1].copy()


def load_gray(path: str | Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def touch_done(job_dir: str | Path) -> None:
    """Write the empty DONE marker; must be the very last action of a job."""
    atomic_write_bytes(Path(job_dir) / "DONE", b"")


def is_done(job_dir: str | Path) -> bool:
    return (Path(job_dir) / "DONE").exists()


def safe_name(s: str) -> str:
    """File-system-safe version of an id (slashes become double underscores)."""
    return s.replace("/", "__").replace(" ", "_")


def read_csv_safe(path: str | Path, **kw) -> pd.DataFrame | None:
    """``pd.read_csv`` that returns None for a missing, empty or header-only file (no crash in reports)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None
