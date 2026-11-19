"""Inference jobs writing the prediction store (one Parquet file per condition, resumable).

Store layout: ``predictions/<eval_dataset>/<train job id with '/' -> '__'>/<condition>.parquet``.
Kinds:
* ``real``    - test images of the fold (or all images): HQ rows get condition 'clean', LQ rows 'real'.
* ``synth``   - HQ test images under every corruption condition.
* ``entropy`` - DDR ungradable images plus a sample of gradable ones (DeepDRiD-trained models).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from realcal.utils.amp import amp_dtype, forward_safe
from realcal.config import Ctx
from realcal.data.common import load_metadata
from realcal.data.datasets import FundusDataset, make_loader
from realcal.data.splits import table_for_fold
from realcal.jobs import Job, job_dir
from realcal.models.adapt import Classifier, setup_mode
from realcal.models.backbones import build_backbone
from realcal.train.threshold import binary_score
from realcal.utils.io import read_json, safe_name, write_parquet
from realcal.utils.signals import check_stop

log = logging.getLogger(__name__)

META_COLS = ["image_id", "patient_id", "eye_id", "label", "grade", "hq", "lq_any", "lq_artifact", "lq_clarity",
             "lq_field", "site", "camera", "ungradable"]


def pred_dir(ctx: Ctx, eval_dataset: str, train_job_id: str) -> Path:
    return ctx.out / "predictions" / eval_dataset / safe_name(train_job_id)


def eval_rows(ctx: Ctx, kind: str, image_set: str, train_params: dict, eval_dataset: str) -> pd.DataFrame:
    """Rows to score (metadata columns + 'condition_base')."""
    if image_set == "test_fold":
        v = "cv" if train_params.get("variant") == "cv" else "main"
        tab = table_for_fold(ctx, eval_dataset, int(train_params["fold"]), v)
        rows = tab[tab["role"] == "test"].copy()
    elif image_set == "all":
        rows = load_metadata(ctx, eval_dataset).copy()
    elif image_set == "ddr_entropy":
        from realcal.models.features import ddr_entropy_ids

        meta = load_metadata(ctx, "ddr")
        rows = meta[meta["image_id"].isin(set(ddr_entropy_ids(ctx)))].copy()
    else:
        raise ValueError(image_set)
    if kind == "synth":
        rows = rows[rows["hq"]]
    return rows.reset_index(drop=True)


def load_model(ctx: Ctx, train_job_id: str, tp: dict, device: torch.device) -> Classifier:
    n_classes = int(ctx.ds_cfg(tp["dataset"]).n_classes)
    bb = build_backbone(ctx, tp["backbone"], int(tp["size"]))
    model = Classifier(bb, n_classes)
    setup_mode(model, tp["mode"], ctx.cfg)
    st = torch.load(ctx.out / "runs" / train_job_id / "best.pt", map_location="cpu", weights_only=False)["model"]
    st = {k.replace("_orig_mod.", ""): v for k, v in st.items()}
    if tp["mode"] == "lp":
        model.head.load_state_dict(st)
    else:
        res = model.load_state_dict(st, strict=False)
        if res.unexpected_keys:
            raise RuntimeError(f"unexpected keys when loading {train_job_id}: {res.unexpected_keys[:5]}")
    model = model.to(device).eval()
    if bb.is_cnn:
        model = model.to(memory_format=torch.channels_last)
    model.amp_dt = amp_dtype(ctx, tp["backbone"])
    return model


@torch.no_grad()
def _probs_images(ctx: Ctx, model: Classifier, dataset: str, ids: list[str], size: int, condition: str,
                  bs: int, device: torch.device) -> np.ndarray:
    bb = model.backbone
    ds = FundusDataset(ctx, dataset, ids, None, size, bb.mean, bb.std, condition=condition, train=False)
    loader = make_loader(ds, bs, shuffle=False, num_workers=int(ctx.extra.get("num_workers", 4)))
    out = np.zeros((len(ids), model.head.out_features), np.float32)
    for x, _, idx in loader:
        check_stop()
        x = x.to(device, non_blocking=True)
        if bb.is_cnn:
            x = x.contiguous(memory_format=torch.channels_last)
        lg = forward_safe(model, x, device, getattr(model, "amp_dt", torch.float16))
        out[idx.numpy()] = torch.softmax(lg, 1).cpu().numpy()
    return out


@torch.no_grad()
def _probs_lp(ctx: Ctx, model: Classifier, tp: dict, dataset: str, ids: list[str], condition: str,
              device: torch.device) -> np.ndarray:
    from realcal.models.features import get_features

    cond = "clean" if condition in ("clean", "real") else condition
    f = get_features(ctx, tp["backbone"], dataset, int(tp["size"]), cond, ids, bb=model.backbone)
    lg = model.head(torch.from_numpy(f).to(device)).float()
    return torch.softmax(lg, 1).cpu().numpy()


def run_predict_job(job: Job, ctx: Ctx) -> None:
    from realcal.corruptions.registry import active_conditions

    p = job.params
    tp, tid = p["train_params"], p["train_job_id"]
    eval_ds, kind = p["eval_dataset"], p["kind"]
    tjd = ctx.out / "runs" / tid
    if not (tjd / "DONE").exists():
        raise RuntimeError(f"training job {tid} is not finished")
    thr = read_json(tjd / "threshold.json")
    rows = eval_rows(ctx, kind, p["image_set"], tp, eval_ds)
    if kind == "synth":
        conds = p.get("conditions") or [c.name for c in active_conditions(ctx)]
        plan = [(c, rows) for c in conds]
    else:
        lq = rows["lq_any"].astype(bool)
        plan = [("clean", rows[~lq]), ("real", rows[lq])]
    out_dir = pred_dir(ctx, eval_ds, tid)
    todo = [(c, r) for c, r in plan if not (out_dir / f"{c}.parquet").exists() and len(r)]
    if not todo:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ctx, tid, tp, device)
    n_classes = model.head.out_features
    neg = int(ctx.cfg.threshold.multiclass_negative_class)
    size = int(tp["size"])
    for cond, r in todo:
        check_stop()
        ids = r["image_id"].tolist()
        if tp["mode"] == "lp":
            probs = _probs_lp(ctx, model, tp, eval_ds, ids, cond, device)
        else:
            probs = _probs_images(ctx, model, eval_ds, ids, size, cond, int(p["batch_size"]), device)
        score = binary_score(probs, n_classes, neg)
        df = r[[c for c in META_COLS if c in r.columns]].copy().reset_index(drop=True)
        df.insert(0, "job_id", tid)
        df["dataset"] = eval_ds
        df["train_dataset"] = tp["dataset"]
        df["variant"] = tp.get("variant", "main")
        df["backbone"] = tp["backbone"]
        df["mode"] = tp["mode"]
        df["family"] = ctx.family(tp["backbone"])
        df["fold"] = int(tp["fold"])
        df["seed"] = int(tp["seed"])
        df["size"] = size
        df["condition"] = cond
        df["prob"] = score.astype(np.float32)
        for k in range(n_classes):
            df[f"prob_{k}"] = probs[:, k]
        df["threshold"] = float(thr["threshold"])
        df["pred_at_threshold"] = (score >= float(thr["threshold"])).astype(np.int8)
        for c in ("site", "camera", "patient_id", "eye_id"):
            if c in df:
                df[c] = df[c].astype("string")
        for c in ("label", "grade"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        write_parquet(out_dir / f"{cond}.parquet", df)
    log.info("%s: wrote %d condition file(s)", job.id, len(todo))
    _ = job_dir(ctx, job.id)
