"""Adaptation modes: linear probe (lp), LoRA / CNN last stage (lora), full fine-tuning (ft).

LoRA (rank 8, alpha 16, dropout 0.1) is applied to the attention *query* and *value* projections.
For timm ViTs the projections are fused in ``qkv``; :class:`FusedQKVLoRA` adds low-rank updates to
the q and v slices only. For the Hugging Face DINOv3 model, ``q_proj`` and ``v_proj`` are wrapped
with :class:`LoRALinear`. CNN backbones replace LoRA with "last stage" unfreezing.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from realcal.models.backbones import Backbone


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.drop = nn.Dropout(dropout)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


class FusedQKVLoRA(nn.Module):
    """LoRA on the q and v thirds of a fused qkv projection (k untouched)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        d = base.out_features // 3
        self.d, self.scale = d, alpha / r
        self.drop = nn.Dropout(dropout)
        self.Aq = nn.Parameter(torch.zeros(r, base.in_features))
        self.Bq = nn.Parameter(torch.zeros(d, r))
        self.Av = nn.Parameter(torch.zeros(r, base.in_features))
        self.Bv = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.Aq, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Av, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:  # some timm code paths read qkv.weight
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        xd = self.drop(x)
        dq = (xd @ self.Aq.t() @ self.Bq.t()) * self.scale
        dv = (xd @ self.Av.t() @ self.Bv.t()) * self.scale
        zeros = torch.zeros_like(dq)
        return out + torch.cat([dq, zeros, dv], dim=-1)


def inject_lora(bb: Backbone, r: int, alpha: float, dropout: float) -> int:
    """Insert LoRA modules in place; returns the number of wrapped projections."""
    n = 0
    if bb.arch == "timm_vit":
        for blk in bb.net.blocks:
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = FusedQKVLoRA(attn.qkv, r, alpha, dropout)
                # timm EVA-style attention (e.g. DINOv3) may call F.linear(x, qkv.weight, bias) directly when
                # it keeps separate q/v biases, which would bypass LoRA; force the module call path instead.
                if getattr(attn, "q_bias", None) is not None and hasattr(attn, "qkv_bias_separate"):
                    attn.qkv_bias_separate = True
                n += 1
            else:
                for nm in ("q", "q_proj"):
                    if hasattr(attn, nm):
                        setattr(attn, nm, LoRALinear(getattr(attn, nm), r, alpha, dropout))
                        n += 1
                for nm in ("v", "v_proj"):
                    if hasattr(attn, nm):
                        setattr(attn, nm, LoRALinear(getattr(attn, nm), r, alpha, dropout))
                        n += 1
    elif bb.arch == "hf_dinov3":
        targets = []
        for name, mod in bb.net.named_modules():
            for child_name in ("q_proj", "v_proj", "query", "value"):
                child = getattr(mod, child_name, None)
                if isinstance(child, nn.Linear):
                    targets.append((mod, child_name, child))
        for mod, child_name, child in targets:
            setattr(mod, child_name, LoRALinear(child, r, alpha, dropout))
            n += 1
    if n == 0:
        raise RuntimeError(f"LoRA: no attention projections found in {bb.key}")
    return n


class Classifier(nn.Module):
    def __init__(self, backbone: Backbone, n_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, n_classes)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def setup_mode(model: Classifier, mode: str, cfg) -> tuple[list[dict], str]:
    """Freeze/unfreeze according to ``mode``; returns optimizer parameter groups and the effective mode name.

    Effective mode names: 'lp', 'lora', 'last_stage' (CNN replacement of lora), 'ft'.
    """
    bb = model.backbone
    lr = float(cfg.train.lr[mode])
    for p in bb.parameters():
        p.requires_grad = False
    eff = mode
    if mode == "lp":
        groups = [{"params": list(model.head.parameters()), "lr": lr}]
    elif mode == "lora":
        if bb.supports_lora and not bb.is_cnn:
            lc = cfg.train.lora
            inject_lora(bb, int(lc.rank), float(lc.alpha), float(lc.dropout))
            for n, p in bb.named_parameters():
                p.requires_grad = any(t in n.split(".")[-1] for t in ("A", "B", "Aq", "Bq", "Av", "Bv")) and \
                    n.split(".")[-1] in ("A", "B", "Aq", "Bq", "Av", "Bv")
        else:
            eff = "last_stage"
            for m in bb.last_stage_modules():
                for p in m.parameters():
                    p.requires_grad = True
        bb_params = [p for p in bb.parameters() if p.requires_grad]
        groups = [{"params": bb_params, "lr": lr}, {"params": list(model.head.parameters()), "lr": lr}]
    elif mode == "ft":
        for p in bb.parameters():
            p.requires_grad = True
        groups = [{"params": [p for p in bb.parameters()], "lr": lr},
                  {"params": list(model.head.parameters()), "lr": lr * float(cfg.train.head_lr_mult_ft)}]
    else:
        raise ValueError(mode)
    for p in model.head.parameters():
        p.requires_grad = True
    return groups, eff


def trainable_state(model: nn.Module) -> dict:
    """State dict restricted to trainable parameters plus all buffers of trainable modules (BN stats)."""
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    keep = {}
    for k, v in sd.items():
        if k in names:
            keep[k] = v
        elif not k.endswith(("weight", "bias")) and any(k.rsplit(".", 1)[0] == n.rsplit(".", 1)[0] for n in names):
            keep[k] = v  # BN running stats of unfrozen stages
    return keep
