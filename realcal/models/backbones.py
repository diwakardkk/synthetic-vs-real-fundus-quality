"""Backbone registry with strictly local weight loading.

Every backbone is wrapped in :class:`Backbone`, which exposes ``forward(x) -> (B, D)`` pooled
embeddings, ``embed_dim``, ``mean``, ``std``, ``supports_lora``, ``arch`` and helpers used by the
adaptation code. ViTs handle 224 and 448 inputs by resampling absolute position embeddings
(DINOv3 uses RoPE and needs no resampling).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from realcal.config import Ctx

log = logging.getLogger(__name__)


class WeightsMissing(FileNotFoundError):
    """Raised when a backbone's local weights are absent (the job is reported and skipped)."""


def weights_path(ctx: Ctx, key: str) -> Path:
    c = ctx.encoder_cfg(key)
    p = Path(str(c.weights))
    return p if p.is_absolute() else ctx.weights_root / p


def weights_available(ctx: Ctx, key: str) -> bool:
    """True only for a real weight file, or a snapshot folder that contains config.json and weights."""
    p = weights_path(ctx, key)
    if p.is_dir():
        return (p / "config.json").exists() and (any(p.glob("*.safetensors")) or any(p.glob("*.bin")))
    return p.is_file() and p.stat().st_size > 1_000_000


def _load_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _clean_state(sd: dict, c) -> dict:
    key = c.get("state_key")
    if key and isinstance(sd, dict) and key in sd:
        sd = sd[key]
    for k in ("model", "state_dict", "teacher"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict) and not any(torch.is_tensor(v) for v in sd.values()):
            sd = sd[k]
    keep = c.get("keep_prefix")
    if keep:
        sd = {k: v for k, v in sd.items() if k.startswith(str(keep))}
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for pre in c.get("strip_prefixes", []) or []:
            if k.startswith(str(pre)):
                k = k[len(str(pre)):]
        if any(k.startswith(str(d)) for d in (c.get("drop_prefixes", []) or [])):
            continue
        for a, b in (c.get("rename", {}) or {}).items():
            if k.startswith(str(a)):
                k = str(b) + k[len(str(a)):]
        out[k] = v
    return out


def _resample_pos(model: nn.Module, sd: dict) -> dict:
    if "pos_embed" not in sd or not hasattr(model, "pos_embed"):
        return sd
    from timm.layers import resample_abs_pos_embed

    tgt = model.pos_embed
    if sd["pos_embed"].shape == tgt.shape:
        return sd
    npt = 0 if getattr(model, "no_embed_class", False) else getattr(model, "num_prefix_tokens", 1)
    grid = model.patch_embed.grid_size
    sd["pos_embed"] = resample_abs_pos_embed(sd["pos_embed"], new_size=list(grid), num_prefix_tokens=npt)
    return sd


class Backbone(nn.Module):
    """Uniform wrapper: pooled features of shape (B, D)."""

    def __init__(self, key: str, net: nn.Module, arch: str, embed_dim: int, mean, std, supports_lora: bool):
        super().__init__()
        self.key, self.net, self.arch = key, net, arch
        self.embed_dim, self.mean, self.std = int(embed_dim), list(mean), list(std)
        self.supports_lora = bool(supports_lora)

    @property
    def is_cnn(self) -> bool:
        return self.arch in ("timm_cnn", "tv_resnet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "hf_dinov3":
            out = self.net(pixel_values=x)
            return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        if self.arch == "tv_resnet":
            return torch.flatten(self.net(x), 1)
        return self.net(x)

    def last_stage_modules(self) -> list[nn.Module]:
        """Final stage (for the CNN 'last stage' adaptation mode)."""
        n = self.net
        if self.arch == "tv_resnet":
            return [n.layer4]
        if hasattr(n, "layer4"):
            return [n.layer4]
        if hasattr(n, "blocks") and hasattr(n, "conv_head"):
            mods = [n.blocks[-1], n.conv_head]
            if hasattr(n, "bn2"):
                mods.append(n.bn2)
            return mods
        if hasattr(n, "stages"):
            mods = [n.stages[-1]]
            if hasattr(n, "norm_pre"):
                mods.append(n.norm_pre)
            return mods
        raise ValueError(f"no last stage known for {self.key}")

    def set_grad_checkpointing(self, on: bool = True) -> None:
        if hasattr(self.net, "set_grad_checkpointing"):
            self.net.set_grad_checkpointing(on)
        elif hasattr(self.net, "gradient_checkpointing_enable") and on:
            self.net.gradient_checkpointing_enable()


def build_backbone(ctx: Ctx, key: str, size: int = 224) -> Backbone:
    """Build backbone ``key`` from local weights only."""
    import timm

    c = ctx.encoder_cfg(key)
    path = weights_path(ctx, key)
    if not weights_available(ctx, key):
        raise WeightsMissing(f"weights for '{key}' not found at {path}; run scripts/download_weights.sh")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    kind = str(c.kind)
    if kind == "timm_hub":
        kw = {}
        name = str(c.timm_name)
        if "vit" in name or "dinov2" in name:
            kw["img_size"] = size
        net = timm.create_model(name, pretrained=True, num_classes=0, pretrained_cfg_overlay={"file": str(path)}, **kw)
        arch = "timm_vit" if hasattr(net, "blocks") and hasattr(net, "patch_embed") else "timm_cnn"
        return Backbone(key, net, arch, net.num_features, c.mean, c.std, bool(c.supports_lora))
    if kind == "timm_ckpt":
        net = timm.create_model(str(c.timm_name), pretrained=False, num_classes=0, img_size=size,
                                global_pool=str(c.get("global_pool", "token")))
        sd = _resample_pos(net, _clean_state(_load_state(path), c))
        res = net.load_state_dict(sd, strict=False)
        n_tot = len(net.state_dict())
        missing = [k for k in res.missing_keys if not k.startswith(("fc_norm", "head"))]
        log.info("%s: loaded %d tensors; missing %d, unexpected %d", key, len(sd), len(res.missing_keys), len(res.unexpected_keys))
        if len(missing) > 0.2 * n_tot:
            raise RuntimeError(f"{key}: checkpoint does not match the architecture ({len(missing)} of {n_tot} "
                               f"tensors missing, e.g. {missing[:5]}). Check state_key/strip_prefixes in configs/backbones.yaml")
        return Backbone(key, net, "timm_vit", net.num_features, c.mean, c.std, bool(c.supports_lora))
    if kind == "tv_resnet":
        import torchvision

        rn = torchvision.models.resnet50(weights=None)
        rn.fc = nn.Identity()
        sd = _clean_state(_load_state(path), c)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        res = rn.load_state_dict(sd, strict=False)
        log.info("%s: missing %d, unexpected %d", key, len(res.missing_keys), len(res.unexpected_keys))
        if len(res.missing_keys) > 0.2 * len(rn.state_dict()):
            raise RuntimeError(f"{key}: checkpoint does not match torchvision ResNet-50 ({len(res.missing_keys)} missing, "
                               f"e.g. {res.missing_keys[:5]}). Check strip_prefixes in configs/backbones.yaml")
        body = nn.Sequential()
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool"):
            body.add_module(name, getattr(rn, name))
        return Backbone(key, body, "tv_resnet", 2048, c.mean, c.std, False)
    if kind == "hf_dinov3":
        from transformers import AutoModel

        net = AutoModel.from_pretrained(str(path), local_files_only=True)
        dim = int(getattr(net.config, "hidden_size", 768))
        return Backbone(key, net, "hf_dinov3", dim, c.mean, c.std, True)
    raise ValueError(f"unknown backbone kind {kind}")
