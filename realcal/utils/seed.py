"""Deterministic seeding.

* ``derive_seed(job_id, seed)`` gives the global seed of a job.
* ``hash_seed(*parts)`` gives the per-image corruption seed from (image id, corruption, severity),
  so every model sees exactly the same corrupted images.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np


def hash_seed(*parts: Any) -> int:
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def derive_seed(job_id: str, seed: int) -> int:
    return hash_seed("job", job_id, seed)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng_for(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(hash_seed(*parts))


def get_rng_states() -> dict:
    import torch

    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_states(st: dict) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except RuntimeError:
            pass
