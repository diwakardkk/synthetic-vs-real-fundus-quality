"""Pre-registration stage-2 lock.

``python -m realcal.cli freeze`` prints the frozen RealCal weights and writes
``prereg/stage2_lock.json`` containing the hash of ``prereg/frozen_weights.json``. Every stage that
reads model predictions on FIVES, MSHF or DDR calls :func:`require_lock` first.
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time

from realcal.config import Ctx
from realcal.utils.io import read_json, write_json

log = logging.getLogger(__name__)


class LockError(RuntimeError):
    pass


def _file_hash(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def freeze(ctx: Ctx, yes: bool = False) -> None:
    fp = ctx.out / "prereg" / "frozen_weights.json"
    lock = ctx.out / "prereg" / "stage2_lock.json"
    if not fp.exists():
        raise LockError(f"{fp} not found: run exp3 first")
    fw = read_json(fp)
    print("=" * 80)
    print("FROZEN RealCal WEIGHTS (these will be applied unchanged to FIVES, MSHF and DDR)")
    print(f"  file           : {fp}")
    print(f"  intercept      : {fw['intercept']:.5f}   lambda: {fw['lambda']}")
    for g, w in zip(fw["groups"], fw["group_weights"]):
        print(f"  group {g:14s}: {w:.5f}")
    nz = sum(1 for w in fw["level1b"]["weights"] if w > 1e-6)
    print(f"  level 1b       : {nz} non-zero condition weights of {len(fw['level1b']['weights'])}")
    top = sorted(zip(fw["level2"]["conditions"], fw["level2"]["pi"]), key=lambda t: -t[1])[:8]
    print("  level 2 top pi : " + ", ".join(f"{c}={p:.3f}" for c, p in top))
    print(f"  code hash      : {fw.get('code_hash')}   config hash: {fw.get('config_hash')}")
    print(f"  smoke run      : {fw.get('smoke')}")
    print("=" * 80)
    if lock.exists():
        old = read_json(lock)
        if old.get("frozen_sha256") == _file_hash(fp):
            print(f"Lock already exists and matches ({lock}).")
            return
        raise LockError(f"A different lock exists at {lock}; refusing to overwrite. Remove it manually only if "
                        "pre-registration stage 2 has not been submitted.")
    if not yes:
        if not sys.stdin.isatty():
            raise LockError("non-interactive session: re-run with --yes to confirm the freeze")
        ans = input("Type FREEZE to create the stage-2 lock: ").strip()
        if ans != "FREEZE":
            print("Not frozen.")
            return
    write_json(lock, {"frozen_sha256": _file_hash(fp), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "frozen_path": str(fp)})
    print(f"Stage-2 lock written: {lock}")


def require_lock(ctx: Ctx) -> dict:
    fp = ctx.out / "prereg" / "frozen_weights.json"
    lock = ctx.out / "prereg" / "stage2_lock.json"
    if not lock.exists() or not fp.exists():
        raise LockError("External-data stages are locked. Run exp3 and then `python -m realcal.cli freeze` first.")
    if read_json(lock).get("frozen_sha256") != _file_hash(fp):
        raise LockError("frozen_weights.json does not match the stage-2 lock (it was modified after freezing).")
    return read_json(fp)
