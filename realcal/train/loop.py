"""Training job: HQ-only training, early stopping on validation AUROC, full checkpoint/resume.

* ``last.pt`` every epoch (trainable weights, optimizer, GradScaler, epoch, best metric, early-stopping
  counter, RNG states); ``best.pt`` on validation improvement. All writes are atomic.
* SIGTERM/SIGINT: after the current batch the state is saved to ``last.pt`` (the interrupted epoch is
  redone on resume) and ``JobInterrupted`` propagates so that the job becomes pending.
* Linear probes train on cached frozen features (no augmentation, tiny GPU footprint).
* After training: HQ-validation threshold at 90 % specificity -> ``threshold.json`` (never recomputed).
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from realcal.config import Ctx
from realcal.data.datasets import FundusDataset, make_loader
from realcal.data.splits import table_for_fold
from realcal.eval.metrics import auroc
from realcal.jobs import Job, job_dir
from realcal.models.adapt import Classifier, setup_mode, trainable_state
from realcal.models.backbones import build_backbone
from realcal.train.threshold import binary_score, binary_target, threshold_at_specificity
from realcal.utils.io import is_done, torch_save, write_json, write_parquet
from realcal.utils.amp import amp_dtype, forward_safe
from realcal.utils.provenance import write_provenance
from realcal.utils.seed import derive_seed, get_rng_states, seed_everything, set_rng_states
from realcal.utils.signals import JobInterrupted, check_stop, stop_requested

log = logging.getLogger(__name__)


class EpochSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch permutation (resumable without generator state)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __iter__(self):
        g = np.random.default_rng(self.seed * 1000 + self.epoch)
        return iter(g.permutation(self.n).tolist())

    def __len__(self) -> int:
        return self.n


def lr_factor(epoch_float: float, warmup: int, total: int) -> float:
    if warmup > 0 and epoch_float < warmup:
        return max(1e-3, epoch_float / warmup)
    prog = (epoch_float - warmup) / max(1e-8, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, prog))))


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=k).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (k * counts)
    return torch.tensor(w, dtype=torch.float32)


def freeze_bn_of_frozen(model: nn.Module) -> None:
    """Keep BatchNorm layers whose parameters are frozen in eval mode (running stats unchanged)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in m.parameters(recurse=False)):
                m.eval()


def split_tables(ctx: Ctx, dataset: str, fold: int, variant: str = "main") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tab = table_for_fold(ctx, dataset, fold, "cv" if variant == "cv" else "main")
    lab = tab["label"].notna()
    train = tab[(tab["role"] == "train") & tab["hq"] & lab].reset_index(drop=True)
    val = tab[(tab["role"] == "val") & tab["hq"] & lab].reset_index(drop=True)
    test = tab[tab["role"] == "test"].reset_index(drop=True)
    return train, val, test


def _metric(y: np.ndarray, probs: np.ndarray, n_classes: int, loss: float) -> float:
    a = auroc(y, probs[:, 1] if n_classes == 2 else probs)
    return a if not np.isnan(a) else -loss


# ----------------------------------------------------------------------------- generic loop

class Trainer:
    def __init__(self, ctx: Ctx, job: Job, jd: Path, model: nn.Module, groups: list[dict], n_classes: int,
                 cw: torch.Tensor, device: torch.device, amp_dt: torch.dtype = torch.float16):
        self.ctx, self.job, self.jd, self.model, self.n_classes, self.device = ctx, job, jd, model, n_classes, device
        tc = ctx.cfg.train
        self.base_lrs = [g["lr"] for g in groups]
        self.opt = torch.optim.AdamW(groups, weight_decay=float(tc.weight_decay))
        self.amp = bool(tc.amp) and device.type == "cuda"
        self.amp_dt = amp_dt
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and amp_dt == torch.float16)
        self.cw = cw.to(device)
        self.max_epochs, self.warmup, self.patience = int(tc.max_epochs), int(tc.warmup_epochs), int(tc.patience)
        self.grad_clip = float(tc.grad_clip)
        self.start_epoch, self.best, self.bad = 0, -float("inf"), 0
        self.best_epoch = -1

    # checkpoint ---------------------------------------------------------------------
    def save_last(self, epoch_next: int) -> None:
        torch_save(self.jd / "last.pt", {
            "model": trainable_state(self.model), "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
            "epoch": epoch_next, "best": self.best, "bad": self.bad, "best_epoch": self.best_epoch,
            "rng": get_rng_states(),
        })

    def try_resume(self) -> None:
        p = self.jd / "last.pt"
        if not p.exists():
            return
        st = torch.load(p, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"], strict=False)
        try:
            self.opt.load_state_dict(st["opt"])
        except ValueError as e:
            log.warning("optimizer state not restored (%s)", e)
        self.scaler.load_state_dict(st["scaler"])
        self.start_epoch, self.best, self.bad = int(st["epoch"]), float(st["best"]), int(st["bad"])
        self.best_epoch = int(st.get("best_epoch", -1))
        set_rng_states(st["rng"])
        log.info("resumed from last.pt at epoch %d (best %.4f, bad %d)", self.start_epoch, self.best, self.bad)

    def set_lr(self, ef: float) -> float:
        f = lr_factor(ef, self.warmup, self.max_epochs)
        for g, b in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = b * f
        return self.opt.param_groups[-1]["lr"]

    # main ---------------------------------------------------------------------------
    def fit(self, train_iter_fn, val_fn, steps_per_epoch: int, accum: int) -> None:
        self.try_resume()
        if self.bad >= self.patience:
            return
        mpath = self.jd / "metrics.jsonl"
        for epoch in range(self.start_epoch, self.max_epochs):
            t0 = time.time()
            self.model.train()
            freeze_bn_of_frozen(self.model)
            tot, n = 0.0, 0
            self.opt.zero_grad(set_to_none=True)
            for step, (x, y) in enumerate(train_iter_fn(epoch)):
                lr = self.set_lr(epoch + step / max(1, steps_per_epoch))
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dt, enabled=self.amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.float(), y, weight=self.cw)
                self.scaler.scale(loss / accum).backward()
                if (step + 1) % accum == 0 or step + 1 == steps_per_epoch:
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], self.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                tot += float(loss.detach()) * len(y)
                n += len(y)
                if stop_requested():
                    self.save_last(epoch)  # the interrupted epoch is redone on resume
                    raise JobInterrupted("stopped during training")
            vy, vp, vloss = val_fn()
            metric = _metric(vy, vp, self.n_classes, vloss)
            improved = metric > self.best
            if improved:
                self.best, self.bad, self.best_epoch = metric, 0, epoch
                torch_save(self.jd / "best.pt", {"model": trainable_state(self.model), "epoch": epoch, "metric": metric})
            else:
                self.bad += 1
            self.save_last(epoch + 1)
            rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "val_loss": vloss, "val_auroc": metric,
                   "lr": self.opt.param_groups[-1]["lr"], "seconds": time.time() - t0,
                   "gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0,
                   "improved": improved}
            with open(mpath, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log.info("epoch %d loss %.4f val_loss %.4f val_auroc %.4f%s", epoch, rec["train_loss"], vloss, metric,
                     " *" if improved else "")
            if self.bad >= self.patience:
                log.info("early stopping at epoch %d (best %d)", epoch, self.best_epoch)
                break


# ----------------------------------------------------------------------------- job entry

def run_train_job(job: Job, ctx: Ctx) -> None:
    t_start = time.time()
    p = job.params
    jd = job_dir(ctx, job.id)
    if is_done(jd):
        return
    ds, backbone, mode, fold, seed = p["dataset"], p["backbone"], p["mode"], int(p["fold"]), int(p["seed"])
    size, variant = int(p["size"]), p.get("variant", "main")
    seed_everything(derive_seed(job.id, seed))
    dcfg = ctx.ds_cfg(ds)
    n_classes = int(dcfg.n_classes)
    train, val, _ = split_tables(ctx, ds, fold, variant)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(f"empty training ({len(train)}) or validation ({len(val)}) set")
    ytr = train["label"].astype(int).to_numpy()
    yva = val["label"].astype(int).to_numpy()
    cw = class_weights(ytr, n_classes) if bool(ctx.cfg.train.class_weighted) else torch.ones(n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(p["batch_size"])
    eff_bs = int(ctx.cfg.train.batch_size)
    accum = max(1, math.ceil(eff_bs / bs))
    nw = int(ctx.extra.get("num_workers", 4))

    bb = build_backbone(ctx, backbone, size)
    model = Classifier(bb, n_classes)
    groups, eff_mode = setup_mode(model, mode, ctx.cfg)
    model = model.to(device)
    if bb.is_cnn:
        model = model.to(memory_format=torch.channels_last)
    log.info("train %s: %d train / %d val HQ images, mode %s (effective %s), batch %d x accum %d",
             job.id, len(train), len(val), mode, eff_mode, bs, accum)

    if mode == "lp":
        from realcal.models.features import get_features

        ftr = torch.from_numpy(get_features(ctx, backbone, ds, size, "clean", train["image_id"].tolist(), bb=bb.to(device).eval()))
        fva = torch.from_numpy(get_features(ctx, backbone, ds, size, "clean", val["image_id"].tolist(), bb=bb.to(device).eval()))
        head = model.head
        sampler = EpochSampler(len(ftr), derive_seed(job.id, seed))
        steps = math.ceil(len(ftr) / bs)
        ytr_t = torch.from_numpy(ytr)

        def train_iter(epoch):
            sampler.set_epoch(epoch)
            idx = np.fromiter(iter(sampler), dtype=np.int64)
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                yield ftr[b], ytr_t[b]

        @torch.no_grad()
        def val_fn():
            head.eval()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lg = head(fva.to(device)).float()
            loss = float(F.cross_entropy(lg, torch.from_numpy(yva).to(device), weight=cw.to(device)))
            head.train()
            return yva, torch.softmax(lg, 1).cpu().numpy(), loss

        trainer = Trainer(ctx, job, jd, head, [{"params": list(head.parameters()), "lr": groups[0]["lr"]}], n_classes, cw, device)
        trainer.fit(train_iter, val_fn, steps, accum)
        model.head.load_state_dict(torch.load(jd / "best.pt", map_location="cpu", weights_only=False)["model"])
        with torch.no_grad():
            lg = model.head(fva.to(device)).float()
        val_probs = torch.softmax(lg, 1).cpu().numpy()
    else:
        if mode == "ft" and bool(ctx.encoder_cfg(backbone).get("grad_checkpointing_ft", False)):
            bb.set_grad_checkpointing(True)
        if bool(ctx.cfg.train.compile):
            model = torch.compile(model)
        aug = None
        if variant == "aug":
            sc = ctx.cfg.sensitivity
            aug = {"prob": float(sc.aug_prob), "ops": list(sc.aug_ops)}
        dtr = FundusDataset(ctx, ds, train["image_id"].tolist(), ytr, size, bb.mean, bb.std, train=True, corruption_aug=aug)
        dva = FundusDataset(ctx, ds, val["image_id"].tolist(), yva, size, bb.mean, bb.std, train=False)
        sampler = EpochSampler(len(dtr), derive_seed(job.id, seed))
        ltr = torch.utils.data.DataLoader(dtr, batch_size=bs, sampler=sampler, num_workers=nw,
                                          pin_memory=device.type == "cuda", persistent_workers=nw > 0, drop_last=False,
                                          prefetch_factor=4 if nw > 0 else None)
        lva = make_loader(dva, 2 * bs, shuffle=False, num_workers=nw)
        steps = len(ltr)
        cl = bb.is_cnn

        def train_iter(epoch):
            sampler.set_epoch(epoch)
            for x, y, _ in ltr:
                if cl:
                    x = x.contiguous(memory_format=torch.channels_last)
                yield x, y

        adt = amp_dtype(ctx, backbone)

        def val_fn():
            return predict_loader(model, lva, device, cl, cw, adt)

        trainer = Trainer(ctx, job, jd, model, groups, n_classes, cw, device, adt)
        trainer.fit(train_iter, val_fn, steps, accum)
        st = torch.load(jd / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        yv, val_probs, _ = predict_loader(model, lva, device, cl, cw, adt)
    finish_training(ctx, job, jd, val, val_probs, n_classes, eff_mode, t_start)


@torch.no_grad()
def predict_loader(model: nn.Module, loader, device: torch.device, channels_last: bool, cw: torch.Tensor | None = None,
                   dtype: torch.dtype = torch.float16):
    model.eval()
    ys, ps, losses, n = [], [], 0.0, 0
    for x, y, _ in loader:
        check_stop()
        x = x.to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        lg = forward_safe(model, x, device, dtype)
        if cw is not None and (y >= 0).all():
            losses += float(F.cross_entropy(lg, y.to(device), weight=cw.to(device), reduction="sum"))
            n += len(y)
        ys.append(y.numpy())
        ps.append(torch.softmax(lg, 1).cpu().numpy())
    model.train()
    return np.concatenate(ys), np.concatenate(ps), losses / max(n, 1)


def finish_training(ctx: Ctx, job: Job, jd: Path, val: pd.DataFrame, val_probs: np.ndarray, n_classes: int,
                    eff_mode: str, t_start: float) -> None:
    neg = int(ctx.cfg.threshold.multiclass_negative_class)
    y = val["label"].astype(int).to_numpy()
    score = binary_score(val_probs, n_classes, neg)
    thr = threshold_at_specificity(binary_target(y, n_classes, neg), score, float(ctx.cfg.threshold.target_specificity))
    thr["effective_mode"] = eff_mode
    thr["val_auroc"] = auroc(y, val_probs[:, 1] if n_classes == 2 else val_probs)
    write_json(jd / "threshold.json", thr)
    vp = pd.DataFrame({"image_id": val["image_id"], "label": y, "score": score})
    for k in range(n_classes):
        vp[f"prob_{k}"] = val_probs[:, k]
    write_parquet(jd / "val_predictions.parquet", vp)
    write_provenance(jd, ctx.cfg, job.params, t_start)
    last = jd / "last.pt"
    if last.exists():
        last.unlink()  # training finished; keep best.pt only (saves disk for full fine-tunes)
    log.info("threshold %.4f (val Se %.3f, Sp %.3f), val AUROC %.4f", thr["threshold"], thr["val_sensitivity"],
             thr["val_specificity"], thr["val_auroc"])
