"""Condition registry: 75 main conditions (15 corruptions x 5 severities, 6 groups) and B2 conditions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from realcal.utils.seed import rng_for


@dataclass(frozen=True)
class Condition:
    name: str          # e.g. 'defocus_s3' or 'b2_pixelate_s2'
    corruption: str    # corruption name within its suite
    op: str            # operator in ops.OPS
    group: str
    severity: int
    param: float
    spatial: bool
    suite: str         # 'main' or 'b2'


def main_conditions(suite_cfg) -> list[Condition]:
    out = []
    for name, c in suite_cfg.corruptions.items():
        for s, p in enumerate(c.params, start=1):
            out.append(Condition(f"{name}_s{s}", str(name), str(name), str(c.group), s, float(p), bool(c.spatial), "main"))
    return out


def b2_conditions(suite_cfg) -> list[Condition]:
    out = []
    for name, c in suite_cfg.b2.items():
        for s, p in enumerate(c.params, start=1):
            out.append(Condition(f"b2_{name}_s{s}", str(name), str(c.op), "b2", s, float(p), bool(c.spatial), "b2"))
    return out


def all_conditions(suite_cfg) -> list[Condition]:
    return main_conditions(suite_cfg) + b2_conditions(suite_cfg)


def smoke_conditions(suite_cfg, n: int) -> list[Condition]:
    """A small, diverse subset (one corruption from different groups, severity 3)."""
    main = main_conditions(suite_cfg)
    seen, out = set(), []
    for c in main:
        if c.severity == 3 and c.group not in seen:
            out.append(c)
            seen.add(c.group)
        if len(out) >= n:
            break
    b2 = [c for c in b2_conditions(suite_cfg) if c.severity == 3][:1]
    return out + b2


def active_conditions(ctx) -> list[Condition]:
    suite = ctx.cfg.corruption_suite
    if ctx.smoke:
        return smoke_conditions(suite, int(ctx.cfg.smoke.n_conditions))
    return all_conditions(suite)


def active_main(ctx) -> list[Condition]:
    return [c for c in active_conditions(ctx) if c.suite == "main"]


def active_b2(ctx) -> list[Condition]:
    return [c for c in active_conditions(ctx) if c.suite == "b2"]


def by_name(ctx) -> dict[str, Condition]:
    return {c.name: c for c in all_conditions(ctx.cfg.corruption_suite)}


def groups(ctx) -> list[str]:
    return [str(g) for g in ctx.cfg.corruption_suite.groups]


def corruption_rng(image_id: str, cond: Condition) -> np.random.Generator:
    """Per-image randomness from hash(image id, corruption, severity)."""
    return rng_for(image_id, f"{cond.suite}:{cond.corruption}", cond.severity)
