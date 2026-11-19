"""Pre-generate corrupted versions of evaluation HQ images (resumable) and the contact sheet.

Output: ``cache/corrupt/<dataset>/<size>/<condition>/<image_id>.png``.
Which images: every HQ image that is a test image in at least one fold (DeepDRiD: all HQ images;
FIVES: HQ official-test images). One job per (dataset, size, corruption) covering its 5 severities.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from realcal.config import Ctx
from realcal.corruptions.ops import run_op
from realcal.corruptions.registry import Condition, active_conditions, by_name, corruption_rng
from realcal.data.preprocess import mask_path, preproc_path
from realcal.jobs import Job
from realcal.utils.io import load_gray, load_rgb, safe_name, save_png
from realcal.utils.signals import check_stop

log = logging.getLogger(__name__)


def corrupt_path(ctx: Ctx, dataset: str, size: int, condition: str, image_id: str) -> Path:
    return ctx.out / "cache" / "corrupt" / dataset / str(size) / condition / f"{safe_name(image_id)}.png"


def eval_hq_ids(ctx: Ctx, dataset: str) -> list[str]:
    """HQ images that are test images in at least one fold."""
    from realcal.data.common import load_metadata
    from realcal.data.splits import load_splits

    meta = load_metadata(ctx, dataset)
    s = load_splits(ctx, dataset)
    roles = [c for c in s.columns if c.startswith("role_f")]
    test_any = s[roles].eq("test").any(axis=1)
    ids = set(s.loc[test_any, "image_id"])
    cvp = ctx.out / "splits" / f"{dataset}_cv_folds.csv"
    if cvp.exists():  # secondary CV variant: every labelled image is a test image once
        ids |= set(pd.read_csv(cvp, dtype={"image_id": str})["image_id"])
    return sorted(meta.loc[meta["hq"] & meta["image_id"].isin(ids), "image_id"].tolist())


def corrupt_jobs(ctx: Ctx, dataset: str, sizes: list[int]) -> list[Job]:
    conds = active_conditions(ctx)
    corrs: dict[str, list[str]] = {}
    for c in conds:
        corrs.setdefault(f"{c.suite}_{c.corruption}", []).append(c.name)
    jobs = []
    for size in sizes:
        for key, names in corrs.items():
            jobs.append(Job(f"corrupt/{dataset}/{size}/{key}", "corrupt-cache", "corrupt",
                            {"dataset": dataset, "size": size, "conditions": names,
                             "out_dirs": [str(ctx.out / "cache" / "corrupt" / dataset / str(size) / n) for n in names]},
                            gpu=False, mem_mb=0))
    return jobs


def run_corrupt_job(job: Job, ctx: Ctx) -> None:
    p = job.params
    ds, size = p["dataset"], int(p["size"])
    lookup = by_name(ctx)
    conds: list[Condition] = [lookup[n] for n in p["conditions"]]
    ids = eval_hq_ids(ctx, ds)
    suite = ctx.cfg.corruption_suite
    n_new = 0
    for iid in ids:
        check_stop()
        todo = [c for c in conds if not corrupt_path(ctx, ds, size, c.name, iid).exists()]
        if not todo:
            continue
        img = load_rgb(preproc_path(ctx, ds, size, iid))
        mask = load_gray(mask_path(ctx, ds, size, iid)) > 0
        for c in todo:
            out = run_op(c.op, img, mask, c.param, corruption_rng(iid, c), c.spatial, suite)
            save_png(corrupt_path(ctx, ds, size, c.name, iid), out)
            n_new += 1
    log.info("%s: wrote %d corrupted images for %d HQ eval images", job.id, n_new, len(ids))


def contact_sheet(ctx: Ctx, dataset: str, size: int = 224, n_examples: int = 2) -> None:
    """15 rows x (clean + 5 severities) for ``n_examples`` images -> figures/supp_corruption_examples.pdf."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from realcal.corruptions.registry import main_conditions

    ids = eval_hq_ids(ctx, dataset)[:n_examples]
    if not ids:
        return
    suite = ctx.cfg.corruption_suite
    conds = main_conditions(suite)
    names = list(suite.corruptions.keys())
    ncol = 6 * len(ids)
    fig, axes = plt.subplots(len(names), ncol, figsize=(ncol * 0.9, len(names) * 0.9))
    for e, iid in enumerate(ids):
        img = load_rgb(preproc_path(ctx, dataset, size, iid))
        mask = load_gray(mask_path(ctx, dataset, size, iid)) > 0
        for r, name in enumerate(names):
            ax = axes[r, e * 6]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if e == 0:
                ax.set_ylabel(name, fontsize=5, rotation=0, ha="right", va="center")
            for s in range(1, 6):
                c = next(cc for cc in conds if cc.corruption == name and cc.severity == s)
                pth = corrupt_path(ctx, dataset, size, c.name, iid)
                out = load_rgb(pth) if pth.exists() else run_op(c.op, img, mask, c.param, corruption_rng(iid, c), c.spatial, suite)
                a = axes[r, e * 6 + s]
                a.imshow(out)
                a.set_xticks([]), a.set_yticks([])
                if r == 0:
                    a.set_title(f"S{s}", fontsize=6)
        axes[0, e * 6].set_title("clean", fontsize=6)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    out = ctx.p("figures", "supp_corruption_examples.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("contact sheet -> %s", out)


def corrupt_cache_complete(ctx: Ctx, dataset: str, size: int, conditions: list[str]) -> pd.DataFrame:
    """Return a table of missing corrupted files (empty if complete)."""
    ids = eval_hq_ids(ctx, dataset)
    miss = [(c, i) for c in conditions for i in ids if not corrupt_path(ctx, dataset, size, c, i).exists()]
    return pd.DataFrame(miss, columns=["condition", "image_id"])



