"""Logging setup: console plus optional per-job file."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

FMT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_realcal", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(FMT))
        h._realcal = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(level)
    for noisy in ("PIL", "matplotlib", "urllib3", "numba", "h5py", "fsspec", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class JobFileLog:
    """Context manager that mirrors all log records to ``<job_dir>/job.log``."""

    def __init__(self, job_dir: str | Path):
        self.path = Path(job_dir) / "job.log"
        self.handler: logging.FileHandler | None = None

    def __enter__(self) -> "JobFileLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(self.path, mode="a")
        self.handler.setFormatter(logging.Formatter(FMT))
        logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, *exc) -> None:
        if self.handler is not None:
            logging.getLogger().removeHandler(self.handler)
            self.handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
