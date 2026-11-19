"""Stage functions that build job lists and hand them to the scheduler.

Each function is idempotent: completed jobs are skipped, so re-running a stage resumes it.
"""
from __future__ import annotations

import logging

from realcal.config import Ctx
from realcal.corruptions.registry import active_conditions
from realcal.jobs import Job, feature_jobs, predict_jobs, train_jobs
from realcal.models.backbones import weights_available
from realcal.scheduler import run_jobs

log = logging.getLogger(__name__)


def available_backbones(ctx: Ctx, keys: list[str] | None = None) -> list[str]:
    keys = keys or ctx.backbone_keys()
    ok = [k for k in keys if weights_available(ctx, k)]
    for k in keys:
        if k not in ok:
            log.error("WEIGHTS MISSING for backbone '%s' -> its jobs are skipped (run scripts/download_weights.sh)", k)
    return ok


def stage_preprocess(ctx: Ctx, dataset: str, sizes: list[int] | None = None) -> None:
    from realcal.data.common import load_metadata
    from realcal.data.preprocess import preprocess_jobs

    meta = load_metadata(ctx, dataset, rebuild=ctx.force)
    jobs = preprocess_jobs(ctx, dataset, meta)
    if sizes:
        for j in jobs:
            j.params["sizes"] = [int(s) for s in sizes]
            j.id = j.id + ("" if sizes == list(ctx.cfg.preprocess.sizes) else f"-s{'-'.join(map(str, sizes))}")
    run_jobs(ctx, "preprocess", jobs)


def stage_splits(ctx: Ctx, dataset: str) -> None:
    from realcal.data.common import load_metadata
    from realcal.data.splits import make_splits, viability

    meta = load_metadata(ctx, dataset)
    make_splits(ctx, dataset, meta)
    if dataset == "deepdrid":
        viability(ctx, meta)


def stage_corrupt(ctx: Ctx, dataset: str, sizes: list[int] | None = None) -> None:
    from realcal.corruptions.cache import contact_sheet, corrupt_jobs

    sizes = sizes or [int(s) for s in ctx.cfg.corruptions_eval_sizes]
    run_jobs(ctx, "corrupt-cache", corrupt_jobs(ctx, dataset, sizes))
    if dataset == "deepdrid":
        try:
            contact_sheet(ctx, dataset)
        except Exception as e:  # noqa: BLE001
            log.warning("contact sheet failed: %s", e)


def condition_chunks(ctx: Ctx, with_synth: bool = True) -> dict[str, list[str]]:
    chunks: dict[str, list[str]] = {"clean": ["clean"]}
    if with_synth:
        for c in active_conditions(ctx):
            chunks.setdefault(f"{c.suite}_{c.corruption}", []).append(c.name)
    return chunks


def stage_features(ctx: Ctx, dataset: str, size: int = 224, encoders: list[str] | None = None,
                   with_synth: bool = True) -> None:
    """Feature caches: LP backbones (+ analysis encoders) on clean images and on corrupted HQ images."""
    if encoders is None:
        encoders = available_backbones(ctx)
        if dataset == "deepdrid" and size == 224:
            encoders = encoders + [str(ctx.cfg.features.level2_encoder)]
    encoders = [e for e in dict.fromkeys(encoders) if weights_available(ctx, e)]
    jobs = feature_jobs(ctx, dataset, encoders, size, condition_chunks(ctx, with_synth), image_set="all")
    run_jobs(ctx, "features", jobs)


def _filter_jobs(ctx: Ctx, dataset: str, backbones=None, modes=None, folds=None, seeds=None, variant="main") -> list[Job]:
    bbs = available_backbones(ctx, backbones)
    if variant == "cv" and folds is None:
        folds = list(range(int(ctx.cfg.splits.n_folds)))
    return train_jobs(ctx, dataset, bbs, modes, folds, seeds, variant)


def stage_train(ctx: Ctx, dataset: str, **kw) -> list[Job]:
    jobs = _filter_jobs(ctx, dataset, **kw)
    run_jobs(ctx, "train", jobs)
    return jobs


def stage_predict(ctx: Ctx, dataset: str, kind: str, eval_dataset: str | None = None, image_set: str = "test_fold",
                  **kw) -> None:
    from realcal.utils.io import is_done

    tj = _filter_jobs(ctx, dataset, **kw)
    done = [j for j in tj if is_done(ctx.out / "runs" / j.id)]
    if len(done) < len(tj):
        log.warning("predict-%s: %d of %d training jobs are not finished; predicting the finished ones",
                    kind, len(tj) - len(done), len(tj))
    conds = [c.name for c in active_conditions(ctx)] if kind == "synth" else None
    stage = f"predict-{kind}"
    run_jobs(ctx, stage, predict_jobs(ctx, done, eval_dataset or dataset, kind, conds, image_set))
