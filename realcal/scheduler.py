"""SQLite job registry and a multi-GPU worker pool (no DDP).

* One worker process per GPU slot, pinned with ``CUDA_VISIBLE_DEVICES`` and started with the
  ``spawn`` method. Workers pull pending jobs with an atomic claim (``BEGIN IMMEDIATE``).
* Several workers may share one GPU (``--jobs-per-gpu`` or automatic, from per-job memory
  estimates with a safety margin).
* CPU-only jobs use the same pool without GPU pinning.
* SIGTERM/SIGINT: the main process forwards the signal; workers finish the current batch, save
  ``last.pt``, mark their job pending and exit.
* CUDA out-of-memory: the batch size is halved and the job requeued (max attempts from config).
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import sqlite3
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from realcal.config import Ctx
from realcal.jobs import Job, job_dir, run_job
from realcal.utils import signals
from realcal.utils.gpu import detect_gpus, log_gpu_table, slots_for
from realcal.utils.io import is_done, touch_done
from realcal.utils.logging import JobFileLog, setup_logging

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, stage TEXT, kind TEXT, params TEXT, gpu INTEGER, mem_mb INTEGER,
  priority INTEGER DEFAULT 0, status TEXT, device TEXT, pid INTEGER, host TEXT,
  start REAL, end REAL, attempts INTEGER DEFAULT 0, error TEXT, batch TEXT
);
CREATE INDEX IF NOT EXISTS idx_status ON jobs(status, batch);
"""


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Registry:
    """Thin wrapper around ``jobs.sqlite`` (WAL mode)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), timeout=120, isolation_level=None)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA busy_timeout=120000")
        self.con.executescript(SCHEMA)

    # -- registration ---------------------------------------------------------------
    def register(self, jobs: list[Job], ctx: Ctx, batch: str) -> None:
        import socket

        host = socket.gethostname()
        self.con.execute("BEGIN IMMEDIATE")
        try:
            for j in jobs:
                jd = ctx.out / "runs" / j.id
                row = self.con.execute("SELECT status, attempts, pid, host FROM jobs WHERE id=?", (j.id,)).fetchone()
                done_on_disk = is_done(jd)
                if ctx.force:
                    _wipe_job(ctx, j)
                    done_on_disk = False
                if row is None:
                    status = "done" if done_on_disk else "pending"
                    self.con.execute(
                        "INSERT INTO jobs(id,stage,kind,params,gpu,mem_mb,priority,status,attempts,batch) "
                        "VALUES(?,?,?,?,?,?,?,?,0,?)",
                        (j.id, j.stage, j.kind, json.dumps(j.params), int(j.gpu), j.mem_mb, j.priority, status, batch))
                    continue
                status, attempts, pid, rhost = row
                if done_on_disk:
                    new = "done"
                elif ctx.force:
                    new, attempts = "pending", 0
                elif status == "running" and (rhost in (None, host)) and not _pid_alive(pid):
                    new = "pending"
                elif status == "failed" and ctx.retry_failed and attempts < int(ctx.cfg.scheduler.max_attempts):
                    new = "pending"
                elif status == "done" and not done_on_disk:
                    new = "pending"
                else:
                    new = status
                # keep a reduced batch size from an earlier OOM retry
                old_params = json.loads(self.con.execute("SELECT params FROM jobs WHERE id=?", (j.id,)).fetchone()[0])
                params = dict(j.params)
                if "batch_size" in old_params and "batch_size" in params and not ctx.force:
                    params["batch_size"] = min(int(old_params["batch_size"]), int(params["batch_size"]))
                self.con.execute(
                    "UPDATE jobs SET stage=?,kind=?,params=?,gpu=?,mem_mb=?,priority=?,status=?,attempts=?,batch=? WHERE id=?",
                    (j.stage, j.kind, json.dumps(params), int(j.gpu), j.mem_mb, j.priority, new, attempts, batch, j.id))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def reset_dead(self) -> int:
        """Reset jobs marked running whose PID is dead (same host) to pending."""
        import socket

        host = socket.gethostname()
        rows = self.con.execute("SELECT id, pid, host FROM jobs WHERE status='running'").fetchall()
        n = 0
        for jid, pid, rhost in rows:
            if rhost in (None, host) and not _pid_alive(pid):
                self.con.execute("UPDATE jobs SET status='pending' WHERE id=? AND status='running'", (jid,))
                n += 1
        return n

    # -- claiming -------------------------------------------------------------------
    def claim(self, batch: str, device: str, pid: int) -> Job | None:
        import socket

        self.con.execute("BEGIN IMMEDIATE")
        try:
            row = self.con.execute(
                "SELECT id, stage, kind, params, gpu, mem_mb, priority FROM jobs "
                "WHERE status='pending' AND batch=? ORDER BY priority DESC, id LIMIT 1", (batch,)).fetchone()
            if row is None:
                self.con.execute("COMMIT")
                return None
            self.con.execute(
                "UPDATE jobs SET status='running', device=?, pid=?, host=?, start=?, end=NULL, attempts=attempts+1 WHERE id=?",
                (device, pid, socket.gethostname(), time.time(), row[0]))
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return Job(row[0], row[1], row[2], json.loads(row[3]), bool(row[4]), row[5], row[6])

    def finish(self, job_id: str, status: str, error: str | None = None, params: dict | None = None) -> None:
        if params is not None:
            self.con.execute("UPDATE jobs SET status=?, end=?, error=?, params=? WHERE id=?",
                             (status, time.time(), error, json.dumps(params), job_id))
        else:
            self.con.execute("UPDATE jobs SET status=?, end=?, error=? WHERE id=?", (status, time.time(), error, job_id))

    def attempts(self, job_id: str) -> int:
        r = self.con.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
        return int(r[0]) if r else 0

    def counts(self, batch: str | None = None) -> dict[str, dict[str, int]]:
        q = "SELECT stage, status, COUNT(*) FROM jobs"
        args: tuple = ()
        if batch:
            q += " WHERE batch=?"
            args = (batch,)
        q += " GROUP BY stage, status"
        out: dict[str, dict[str, int]] = {}
        for stage, status, n in self.con.execute(q, args):
            out.setdefault(stage, {})[status] = n
        return out

    def rows(self, where: str = "", args: tuple = ()) -> list[tuple]:
        return self.con.execute(f"SELECT id, stage, status, device, pid, start, end, attempts, error FROM jobs {where}", args).fetchall()

    def close(self) -> None:
        self.con.close()


def _wipe_job(ctx: Ctx, j: Job) -> None:
    jd = ctx.out / "runs" / j.id
    if jd.exists():
        shutil.rmtree(jd, ignore_errors=True)
    for key in ("pred_dir", "out_dir"):
        p = j.params.get(key)
        if p and Path(p).exists():
            shutil.rmtree(p, ignore_errors=True)
    for d in j.params.get("out_dirs") or []:
        if Path(d).exists():
            shutil.rmtree(d, ignore_errors=True)
    if j.kind == "predict" and j.params.get("pred_files"):
        pdir = ctx.out / "predictions" / j.params["eval_dataset"] / j.params["train_job_id"].replace("/", "__")
        for fn in j.params["pred_files"]:
            fp = pdir / fn
            if fp.exists():
                fp.unlink()


def registry(ctx: Ctx) -> Registry:
    return Registry(ctx.out / "jobs.sqlite")


# ----------------------------------------------------------------------------- worker

def _is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return "out of memory" in str(e).lower()


def _worker_main(db_path: str, batch: str, gpu_index: int | None, cfg_dict: dict, ctx_kwargs: dict,
                 num_workers: int) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # CUDA_VISIBLE_DEVICES then uses nvidia-smi numbering
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import matplotlib

    matplotlib.use("Agg")
    setup_logging()
    signals.install_handlers()
    ctx = Ctx(OmegaConf.create(cfg_dict), **ctx_kwargs)
    ctx.extra["num_workers"] = num_workers
    reg = Registry(db_path)
    device = f"gpu{gpu_index}" if gpu_index is not None else "cpu"
    try:
        import torch

        if gpu_index is not None:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.set_num_threads(max(1, num_workers))
    except ImportError:
        pass
    max_attempts = int(ctx.cfg.scheduler.max_attempts)
    while not signals.stop_requested():
        job = reg.claim(batch, device, os.getpid())
        if job is None:
            break
        jd = job_dir(ctx, job.id)
        t0 = time.time()
        with JobFileLog(jd):
            log.info("START %s on %s (attempt %d)", job.id, device, reg.attempts(job.id))
            try:
                run_job(job, ctx)
                if not is_done(jd):
                    touch_done(jd)
                reg.finish(job.id, "done")
                log.info("DONE %s in %.1f s", job.id, time.time() - t0)
            except signals.JobInterrupted:
                reg.finish(job.id, "pending", "interrupted")
                log.warning("INTERRUPTED %s: state saved, job set to pending", job.id)
                break
            except BaseException as e:  # noqa: BLE001
                tb = traceback.format_exc()
                if _is_oom(e) and "batch_size" in job.params:
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    p = dict(job.params)
                    p["batch_size"] = max(1, int(p["batch_size"]) // 2)
                    status = "pending" if reg.attempts(job.id) < max_attempts else "failed"
                    reg.finish(job.id, status, f"CUDA OOM; batch size -> {p['batch_size']}", params=p)
                    log.error("OOM in %s: batch size halved to %d, status %s", job.id, p["batch_size"], status)
                else:
                    reg.finish(job.id, "failed", tb[-4000:])
                    log.error("FAILED %s:\n%s", job.id, tb)
                if isinstance(e, KeyboardInterrupt):
                    break
    reg.close()


# ----------------------------------------------------------------------------- main API

def run_jobs(ctx: Ctx, stage: str, jobs: list[Job]) -> dict[str, int]:
    """Register ``jobs`` and execute every pending one; returns status counts for this batch."""
    if not jobs:
        log.info("[%s] nothing to do", stage)
        return {}
    reg = registry(ctx)
    n_reset = reg.reset_dead()
    if n_reset:
        log.info("Reset %d stale running job(s) to pending", n_reset)
    batch = f"{stage}-{uuid.uuid4().hex[:8]}"
    reg.register(jobs, ctx, batch)
    pending = [r for r in reg.rows("WHERE batch=? AND status='pending'", (batch,))]
    done = len(reg.rows("WHERE batch=? AND status='done'", (batch,)))
    log.info("[%s] %d jobs: %d done already, %d pending", stage, len(jobs), done, len(pending))
    if not pending:
        reg.close()
        return _summary(ctx, batch, stage)

    use_gpu = any(j.gpu for j in jobs)
    cpu_count = os.cpu_count() or 4
    if use_gpu:
        gpus = detect_gpus(ctx.gpus)
        log_gpu_table(gpus)
        pend_ids = {r[0] for r in pending}
        mem = max([j.mem_mb for j in jobs if j.id in pend_ids] or [2000])
        if not gpus:
            slots = {None: 1}
        elif ctx.jobs_per_gpu:
            slots = {g.index: int(ctx.jobs_per_gpu) for g in gpus}
        else:
            slots = slots_for(gpus, mem, float(ctx.cfg.scheduler.safety_margin), int(ctx.cfg.scheduler.max_jobs_per_gpu))
        workers = [g for g, n in slots.items() for _ in range(n)]
        log.info("[%s] per-job memory estimate %d MB -> worker slots %s", stage, mem, slots)
    else:
        n = int(ctx.cfg.scheduler.cpu_workers)
        n = cpu_count if n <= 0 else n
        workers = [None] * max(1, min(n, len(pending)))
    workers = workers[: max(1, len(pending))] if len(pending) < len(workers) else workers
    nw = max(1, min(int(ctx.cfg.train.num_workers_max), cpu_count // max(1, len(workers)))) if use_gpu else 0
    cfg_dict = OmegaConf.to_container(ctx.cfg, resolve=True)
    ctx_kwargs = {"smoke": ctx.smoke, "force": False, "retry_failed": False, "gpus": ctx.gpus,
                  "jobs_per_gpu": ctx.jobs_per_gpu}
    reg.close()

    spawn = mp.get_context("spawn")
    procs = []
    for g in workers:
        p = spawn.Process(target=_worker_main, args=(str(ctx.out / "jobs.sqlite"), batch, g, cfg_dict, ctx_kwargs, nw),
                          daemon=False)
        p.start()
        procs.append(p)
    log.info("[%s] started %d worker(s)", stage, len(procs))

    def _forward(signum, frame):  # noqa: ARG001
        log.warning("Signal %d received: asking workers to stop after the current batch", signum)
        signals.request_stop()
        for p in procs:
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    old_term = signal.signal(signal.SIGTERM, _forward)
    old_int = signal.signal(signal.SIGINT, _forward)
    try:
        for p in procs:
            p.join()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return _summary(ctx, batch, stage)


def _summary(ctx: Ctx, batch: str, stage: str) -> dict[str, int]:
    reg = registry(ctx)
    c = reg.counts(batch).get(stage, {})
    failed = reg.rows("WHERE batch=? AND status='failed'", (batch,))
    reg.close()
    log.info("[%s] summary: %s", stage, c)
    for r in failed[:20]:
        last = (r[8] or "").strip().splitlines()[-1:] or [""]
        log.error("[%s] failed job %s: %s", stage, r[0], last[0])
    if failed:
        log.error("[%s] %d failed job(s). Re-run with --retry-failed after fixing the cause.", stage, len(failed))
    return c


def run_inline(ctx: Ctx, stage: str, job_id: str, fn: Callable[[], None], rerun: bool = True) -> bool:
    """Run a CPU analysis step in this process with registry tracking and a DONE marker.

    Analysis steps are cheap and read only the prediction store, so by default they are re-run on every
    call (expensive intermediate results are cached by fingerprint of the prediction store).
    """
    reg = registry(ctx)
    jd = job_dir(ctx, job_id)
    if (rerun or ctx.force) and (jd / "DONE").exists():
        (jd / "DONE").unlink()
    if is_done(jd):
        log.info("[%s] %s already done (use --force to redo)", stage, job_id)
        reg.close()
        return True
    batch = f"{stage}-{uuid.uuid4().hex[:8]}"
    reg.register([Job(job_id, stage, "inline", {}, gpu=False)], ctx, batch)
    reg.claim(batch, "cpu", os.getpid())
    t0 = time.time()
    ok = True
    with JobFileLog(jd):
        try:
            fn()
            touch_done(jd)
            reg.finish(job_id, "done")
            log.info("[%s] %s done in %.1f s", stage, job_id, time.time() - t0)
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            reg.finish(job_id, "failed", tb[-4000:])
            log.error("[%s] %s failed:\n%s", stage, job_id, tb)
            ok = False
    reg.close()
    return ok
