"""Mixed-precision dtype per backbone.

fp16 autocast overflows for some CNNs under full fine-tuning (EfficientNet: activations > 65504 ->
inf -> NaN probabilities). Such backbones set ``amp_dtype: bfloat16`` in backbones.yaml (same
exponent range as fp32; native on Ampere and newer). bf16 needs no GradScaler.
"""
from __future__ import annotations

import torch

from realcal.config import Ctx


def amp_dtype(ctx: Ctx, backbone: str) -> torch.dtype:
    name = str(ctx.encoder_cfg(backbone).get("amp_dtype", ctx.cfg.train.get("amp_dtype", "float16")))
    if name in ("bfloat16", "bf16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def forward_safe(model, x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Autocast forward pass; a batch with non-finite logits is recomputed in fp32."""
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        lg = model(x).float()
    if device.type == "cuda" and not torch.isfinite(lg).all():
        lg = model(x.float()).float()
    return lg
