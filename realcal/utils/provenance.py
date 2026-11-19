"""Provenance: resolved config, git hash, package versions, host, GPU, wall time."""
from __future__ import annotations

import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from realcal.config import CODE_ROOT
from realcal.utils.io import atomic_write_text, read_json, write_json

PACKAGES = ["torch", "torchvision", "timm", "peft", "transformers", "numpy", "pandas", "scipy",
            "scikit-learn", "statsmodels", "cvxpy", "opencv-python-headless", "pyarrow", "omegaconf"]


def git_hash() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture_output=True,
                              text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def code_hash() -> str:
    """Hash of all Python and YAML source files (works without git)."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted(list(CODE_ROOT.rglob("*.py")) + list(CODE_ROOT.rglob("*.yaml"))):
        if "/." in str(p):
            continue
        h.update(p.relative_to(CODE_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def package_versions() -> dict[str, str]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "absent"
    return out


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def write_provenance(job_dir: str | Path, cfg: Any, params: dict, start_time: float) -> None:
    job_dir = Path(job_dir)
    cfg_c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    atomic_write_text(job_dir / "config.yaml", yaml.safe_dump({"params": params, "config": cfg_c}, sort_keys=False))
    write_json(job_dir / "provenance.json", {
        "git_commit": git_hash(), "code_hash": code_hash(), "packages": package_versions(),
        "python": platform.python_version(), "hostname": socket.gethostname(), "gpu": gpu_name(),
        "start": start_time, "end": time.time(), "wall_seconds": time.time() - start_time,
    })


def gpu_hours(out_root: str | Path) -> dict:
    """Aggregate GPU hours over all job directories with provenance.json."""
    tot = 0.0
    by_stage: dict[str, float] = {}
    for p in Path(out_root, "runs").rglob("provenance.json"):
        try:
            d = read_json(p)
        except Exception:  # noqa: BLE001
            continue
        if not d.get("gpu"):
            continue
        h = float(d.get("wall_seconds", 0)) / 3600.0
        stage = p.relative_to(Path(out_root, "runs")).parts[0]
        by_stage[stage] = by_stage.get(stage, 0.0) + h
        tot += h
    return {"total_gpu_hours": tot, "by_stage": by_stage}
