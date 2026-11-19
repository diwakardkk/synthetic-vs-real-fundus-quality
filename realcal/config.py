"""Configuration loading (OmegaConf, no working-directory changes).

All YAML files under ``configs/`` are merged into one tree:
``paths.yaml`` + ``default.yaml`` + ``backbones.yaml`` + ``corruptions.yaml`` and every
``datasets/*.yaml`` under the ``datasets`` key. Command-line overrides use dotted keys.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

CODE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = CODE_ROOT / "configs"


def load_config(overrides: Iterable[str] = (), smoke: bool = False, config_dir: Path | None = None) -> DictConfig:
    """Load and merge the full configuration tree.

    Args:
        overrides: dotted ``key=value`` strings (OmegaConf dotlist syntax).
        smoke: apply the ``smoke`` block (few backbones, 1 fold, 2 epochs, tiny data).
        config_dir: alternative configuration directory.
    """
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    parts = [OmegaConf.load(cdir / n) for n in ("paths.yaml", "default.yaml", "backbones.yaml")]
    corr = OmegaConf.load(cdir / "corruptions.yaml")
    cfg = OmegaConf.merge(*parts, OmegaConf.create({"corruption_suite": corr}))
    ds = {}
    for p in sorted((cdir / "datasets").glob("*.yaml")):
        d = OmegaConf.load(p)
        ds[str(d.name)] = d
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"datasets": ds}))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.set_struct(cfg, False)
    cfg.smoke_mode = bool(smoke)
    if smoke:
        s = cfg.smoke
        cfg.train.max_epochs = int(s.max_epochs)
        cfg.train.warmup_epochs = int(s.warmup_epochs)
        cfg.train.seeds = list(s.seeds)
        cfg.train.patience = 2
        cfg.stats.n_boot = int(s.n_boot)
        cfg.stats.label_noise_reps = int(s.label_noise_reps)
        cfg.calibration.level1b_bootstrap = int(s.level1b_bootstrap)
        cfg.power.n_sim = int(s.power_n_sim)
        cfg.preprocess.sizes = [224]
        cfg.splits.n_folds = 2
    return cfg


def config_hash(obj: Any) -> str:
    """Stable short hash of a (sub)config or JSON-serialisable object."""
    if isinstance(obj, DictConfig):
        obj = OmegaConf.to_container(obj, resolve=True)
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass
class Ctx:
    """Run context shared by all stages."""

    cfg: DictConfig
    smoke: bool = False
    force: bool = False
    retry_failed: bool = False
    gpus: list[int] | None = None
    jobs_per_gpu: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def data_root(self) -> Path:
        return Path(str(self.cfg.data_root))

    @property
    def weights_root(self) -> Path:
        return Path(str(self.cfg.weights_root))

    @property
    def out(self) -> Path:
        base = Path(str(self.cfg.out_root))
        return base / "smoke" if self.smoke else base

    @property
    def base_out(self) -> Path:
        return Path(str(self.cfg.out_root))

    def p(self, *parts: str) -> Path:
        """Path under the (smoke-aware) output root; parent directories are created."""
        path = self.out.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def d(self, *parts: str) -> Path:
        """Directory under the output root (created)."""
        path = self.out.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ds_cfg(self, name: str) -> DictConfig:
        return self.cfg.datasets[name]

    def backbone_keys(self) -> list[str]:
        if self.smoke:
            return list(self.cfg.smoke.backbones)
        return list(self.cfg.backbones.keys())

    def encoder_cfg(self, key: str) -> DictConfig:
        if key in self.cfg.backbones:
            return self.cfg.backbones[key]
        return self.cfg.analysis_encoders[key]

    def folds(self, dataset: str) -> list[int]:
        if self.smoke:
            return list(self.cfg.smoke.folds)
        if dataset == "deepdrid":
            return list(range(int(self.cfg.splits.n_folds)))
        return [0]

    def seeds(self) -> list[int]:
        return [int(s) for s in self.cfg.train.seeds]

    def family(self, backbone: str) -> str:
        return str(self.encoder_cfg(backbone).family)
