"""GPU detection (nvidia-smi + torch) and memory-based slot planning."""
from __future__ import annotations

import logging
import os
import subprocess

# make CUDA device numbers match nvidia-smi (CUDA defaults to FASTEST_FIRST on mixed-GPU machines)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int
    util: int = 0
    compute_cap: str = ""


def query_nvidia_smi() -> list[GPUInfo]:
    """Return physical GPUs as reported by nvidia-smi (empty list if unavailable)."""
    out = None
    for fields in ("index,name,memory.total,memory.free,utilization.gpu,compute_cap",
                   "index,name,memory.total,memory.free,utilization.gpu"):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30, check=True).stdout
            break
        except (OSError, subprocess.SubprocessError):
            continue
    if out is None:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(GPUInfo(int(parts[0]), parts[1], int(float(parts[2])), int(float(parts[3])),
                                int(float(parts[4])) if parts[4].replace(".", "").isdigit() else 0,
                                parts[5] if len(parts) > 5 else ""))
        except ValueError:
            continue
    return gpus


def detect_gpus(override: list[int] | None = None) -> list[GPUInfo]:
    """Detect usable GPUs, honouring CUDA_VISIBLE_DEVICES and an explicit override list.

    Returned indices are *physical* indices (used to set CUDA_VISIBLE_DEVICES in workers).
    """
    phys = query_nvidia_smi()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() != "":
        allowed = [int(x) for x in visible.split(",") if x.strip().isdigit()]
        phys = [g for g in phys if g.index in allowed]
    if override:
        phys = [g for g in phys if g.index in override]
    phys = [g for g in phys if torch_supports(g)]
    if not phys:
        try:
            import torch

            n = torch.cuda.device_count()
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                total = int(p.total_memory / 2**20)
                phys.append(GPUInfo(i, p.name, total, total))
            if override:
                phys = [g for g in phys if g.index in override]
        except Exception:  # noqa: BLE001
            pass
    return phys


def torch_supports(g: GPUInfo) -> bool:
    """False if the installed PyTorch has no kernels for this GPU's architecture (e.g. V100 = sm_70 with
    recent wheels): such a GPU fails every job with cudaErrorNoKernelImageForDevice, so it is skipped."""
    cc = g.compute_cap.replace(".", "").strip()
    if not cc.isdigit():
        return True
    try:
        import torch

        archs = torch.cuda.get_arch_list()
    except Exception:  # noqa: BLE001
        return True
    if not archs:
        return True
    sm = {int(a.split("_")[1]) for a in archs if a.startswith("sm_") and a.split("_")[1].isdigit()}
    ptx = {int(a.split("_")[1]) for a in archs if a.startswith("compute_") and a.split("_")[1].isdigit()}
    ok = int(cc) in sm or any(int(cc) >= p for p in ptx) or any(s // 10 == int(cc) // 10 and s <= int(cc) for s in sm)
    if not ok:
        log.warning("Skipping GPU %d (%s, compute capability %s): not supported by this PyTorch build %s",
                    g.index, g.name, g.compute_cap, archs)
    return ok


def log_gpu_table(gpus: list[GPUInfo]) -> None:
    if not gpus:
        log.warning("No GPU detected: GPU stages will run on CPU (slow).")
        return
    log.info("Detected %d GPU(s):", len(gpus))
    log.info("  %-4s %-32s %10s %10s", "idx", "name", "total MB", "free MB")
    for g in gpus:
        log.info("  %-4d %-32s %10d %10d", g.index, g.name, g.total_mb, g.free_mb)


def slots_for(gpus: list[GPUInfo], job_mem_mb: int, margin: float, max_per_gpu: int) -> dict[int, int]:
    """Number of concurrent workers per GPU that fit the per-job memory estimate."""
    slots = {}
    for g in gpus:
        usable = g.free_mb * (1.0 - margin)
        n = int(usable // max(job_mem_mb, 1))
        slots[g.index] = max(1, min(n, max_per_gpu))
    return slots
