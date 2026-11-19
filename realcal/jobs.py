"""Job definitions, the deterministic id scheme, and job builders for the GPU/CPU stages.

Id scheme (``/`` separated, also the job directory below ``OUT_ROOT/runs``):

* ``preprocess/<dataset>/shard<k>``
* ``corrupt/<dataset>/<size>/<corruption>``
* ``features/<dataset>/<encoder>/<size>/<chunk>``
* ``train/<dataset>[-<variant>]/<backbone>/<mode>/fold<f>/seed<s>``
* ``predict/<eval_dataset>/<kind>/<train job id without the 'train/' prefix>``
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from realcal.config import Ctx
from realcal.utils.io import safe_name

KIND_FUNCS = {
    "preprocess": "realcal.data.preprocess:run_preprocess_job",
    "corrupt": "realcal.corruptions.cache:run_corrupt_job",
    "features": "realcal.models.features:run_features_job",
    "train": "realcal.train.loop:run_train_job",
    "predict": "realcal.eval.predict:run_predict_job",
}


@dataclass
class Job:
    id: str
    stage: str
    kind: str
    params: dict = field(default_factory=dict)
    gpu: bool = True
    mem_mb: int = 2000
    priority: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def job_dir(ctx: Ctx, job_id: str) -> Path:
    d = ctx.out / "runs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve(kind: str) -> Callable[[Job, Ctx], Any]:
    mod, fn = KIND_FUNCS[kind].split(":")
    return getattr(importlib.import_module(mod), fn)


def run_job(job: Job, ctx: Ctx) -> None:
    resolve(job.kind)(job, ctx)


# ----------------------------------------------------------------------------- ids

def train_id(dataset: str, backbone: str, mode: str, fold: int, seed: int, variant: str = "main") -> str:
    ds = dataset if variant == "main" else f"{dataset}-{variant}"
    return f"train/{ds}/{backbone}/{mode}/fold{fold}/seed{seed}"


def parse_train_id(tid: str) -> dict:
    _, ds, backbone, mode, fold, seed = tid.split("/")
    dataset, _, variant = ds.partition("-")
    return {"dataset": dataset, "variant": variant or "main", "backbone": backbone, "mode": mode,
            "fold": int(fold[4:]), "seed": int(seed[4:])}


def predict_id(eval_dataset: str, kind: str, train_job_id: str) -> str:
    return f"predict/{eval_dataset}/{kind}/{train_job_id.removeprefix('train/')}"


def variant_size(variant: str) -> int:
    return 448 if variant == "s448" else 224


# ----------------------------------------------------------------------------- memory

def mem_estimate(ctx: Ctx, backbone: str, mode: str, size: int) -> int:
    m = int(ctx.encoder_cfg(backbone).mem_mb[mode])
    return int(m * (3.5 if size >= 448 else 1.0))


# ----------------------------------------------------------------------------- builders

def train_jobs(ctx: Ctx, dataset: str, backbones: list[str] | None = None, modes: list[str] | None = None,
               folds: list[int] | None = None, seeds: list[int] | None = None, variant: str = "main") -> list[Job]:
    backbones = backbones or ctx.backbone_keys()
    modes = modes or list(ctx.cfg.train.modes)
    folds = folds if folds is not None else ctx.folds(dataset)
    seeds = seeds if seeds is not None else ctx.seeds()
    size = variant_size(variant)
    jobs = []
    for b in backbones:
        for m in modes:
            for f in folds:
                for s in seeds:
                    tid = train_id(dataset, b, m, f, s, variant)
                    jobs.append(Job(tid, "train", "train",
                                    {"dataset": dataset, "backbone": b, "mode": m, "fold": f, "seed": s,
                                     "size": size, "variant": variant,
                                     "batch_size": int(ctx.cfg.train.batch_size)},
                                    gpu=True, mem_mb=mem_estimate(ctx, b, "lp" if m == "lp" else m, size),
                                    priority=10 if m == "ft" else 0))
    return jobs


def predict_jobs(ctx: Ctx, train_job_list: list[Job], eval_dataset: str, kind: str,
                 conditions: list[str] | None = None, image_set: str = "test_fold") -> list[Job]:
    """kind: 'real' (clean HQ + real LQ), 'synth' (corrupted HQ), 'entropy' (DDR ungradable)."""
    jobs = []
    for tj in train_job_list:
        p = dict(tj.params)
        jobs.append(Job(predict_id(eval_dataset, kind, tj.id), f"predict-{kind}", "predict",
                        {"train_job_id": tj.id, "train_params": p, "eval_dataset": eval_dataset, "kind": kind,
                         "conditions": conditions, "image_set": image_set,
                         "batch_size": 2 * int(ctx.cfg.train.batch_size),
                         "pred_files": ([f"{c}.parquet" for c in (conditions or [])] if kind == "synth"
                                        else ["clean.parquet", "real.parquet"]),
                         "pred_subdir": safe_name(tj.id)},
                        gpu=True, mem_mb=mem_estimate(ctx, p["backbone"], "lp", p["size"]) + 1500,
                        priority=5 if p["mode"] == "ft" else 0))
    return jobs


def feature_jobs(ctx: Ctx, dataset: str, encoders: list[str], size: int, chunks: dict[str, list[str]],
                 image_set: str) -> list[Job]:
    """chunks: chunk name -> list of conditions ('clean' or corruption condition names)."""
    jobs = []
    for enc in encoders:
        for cname, conds in chunks.items():
            jobs.append(Job(f"features/{dataset}/{enc}/{size}/{cname}", "features", "features",
                            {"dataset": dataset, "encoder": enc, "size": size, "conditions": conds,
                             "image_set": image_set, "batch_size": int(ctx.cfg.features.batch_size)},
                            gpu=True, mem_mb=mem_estimate(ctx, enc, "lp", size) + 1000))
    return jobs
