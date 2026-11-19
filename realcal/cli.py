"""Single entry point: ``python -m realcal.cli <stage> [options]``.

Every stage is resumable: re-running the same command continues where it stopped.
Launch long stages through ``scripts/launch.sh`` so that they survive SSH disconnects.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from realcal.config import Ctx, load_config  # noqa: E402
from realcal.utils.logging import setup_logging  # noqa: E402

log = logging.getLogger("realcal")

EXTERNAL = {"fives", "mshf", "ddr"}
STAGES = ["inspect", "eda", "preprocess", "fetch-weights", "splits", "features", "train", "predict-real", "corrupt-cache",
          "predict-synth", "exp1", "exp2", "exp3", "freeze", "exp4-fives", "exp4-mshf", "exp4-ddr", "exp4-ddr-repl",
          "clinical", "sensitivity", "power", "report", "status", "all-dev"]


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m realcal.cli", description="RealCal experiment pipeline")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--dataset", default="deepdrid", choices=["deepdrid", "fives", "mshf", "ddr", "all"],
                   help="'all' is only valid for the eda stage")
    p.add_argument("--sizes", type=int, nargs="+", default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--backbones", nargs="+", default=None)
    p.add_argument("--modes", nargs="+", default=None, choices=["lp", "lora", "ft"])
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--variant", default="main", choices=["main", "s448", "aug", "cv"])
    p.add_argument("--which", nargs="+", default=None, help="sensitivity: 448 aug sens perfold subgroup seedsd noise")
    p.add_argument("--tables", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--values", action="store_true")
    p.add_argument("--gpus", default=None, help="comma-separated GPU indices, e.g. 0,1,3")
    p.add_argument("--jobs-per-gpu", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--yes", action="store_true", help="freeze: confirm non-interactively")
    p.add_argument("--config-override", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args(argv)


def make_ctx(a: argparse.Namespace) -> Ctx:
    cfg = load_config(a.config_override, smoke=a.smoke)
    gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    return Ctx(cfg, smoke=a.smoke, force=a.force, retry_failed=a.retry_failed, gpus=gpus, jobs_per_gpu=a.jobs_per_gpu)


def train_kw(a, ctx: Ctx) -> dict:
    return {"backbones": a.backbones, "modes": a.modes, "folds": a.folds, "seeds": a.seeds, "variant": a.variant}


def guard_external(ctx: Ctx, dataset: str, stage: str) -> None:
    if dataset in EXTERNAL and stage not in ("inspect", "preprocess", "splits", "corrupt-cache"):
        from realcal.prereg import require_lock

        require_lock(ctx)


# ----------------------------------------------------------------------------- composite stages

def run_analysis(ctx: Ctx, stage: str, fn) -> bool:
    from realcal.scheduler import run_inline

    return run_inline(ctx, stage, f"analysis/{stage}", fn)


def exp4_fives(ctx: Ctx) -> None:
    from realcal import pipeline as P
    from realcal.experiments.exp4_external import run_fives_analysis

    P.stage_preprocess(ctx, "fives", [224])
    P.stage_splits(ctx, "fives")
    P.stage_corrupt(ctx, "fives")
    P.stage_features(ctx, "fives")
    P.stage_train(ctx, "fives")
    P.stage_predict(ctx, "fives", "real")
    P.stage_predict(ctx, "fives", "synth")
    from realcal.data.common import load_metadata
    from realcal.data.splits import load_splits

    m = load_metadata(ctx, "fives").merge(load_splits(ctx, "fives")[["image_id", "role_f0"]], on="image_id")
    n_lq = int((m["lq_any"] & (m["role_f0"] == "test")).sum())
    if n_lq < int(ctx.cfg.external.fives_min_lq_test):
        log.warning("FIVES official test has only %d LQ images: running the secondary 5-fold CV variant", n_lq)
        from realcal.data.splits import cv_splits

        cv_splits(ctx, "fives")
        P.stage_corrupt(ctx, "fives")
        P.stage_features(ctx, "fives")
        P.stage_train(ctx, "fives", variant="cv")
        P.stage_predict(ctx, "fives", "real", variant="cv")
        P.stage_predict(ctx, "fives", "synth", variant="cv")
    run_analysis(ctx, "exp4-fives", lambda: run_fives_analysis(ctx))


def exp4_mshf(ctx: Ctx) -> None:
    from realcal import pipeline as P
    from realcal.experiments.exp4_external import run_mshf_analysis

    P.stage_preprocess(ctx, "mshf", [224])
    P.stage_splits(ctx, "mshf")
    encs = [str(ctx.cfg.features.level2_encoder), str(ctx.cfg.features.qdet_encoder)]
    P.stage_features(ctx, "mshf", encoders=encs, with_synth=False)
    run_analysis(ctx, "exp4-mshf", lambda: run_mshf_analysis(ctx))


def exp4_ddr(ctx: Ctx) -> None:
    from realcal import pipeline as P
    from realcal.experiments.exp4_external import run_ddr_analysis

    P.stage_preprocess(ctx, "ddr", [224])
    P.stage_splits(ctx, "ddr")
    P.stage_features(ctx, "ddr", encoders=[str(ctx.cfg.features.qdet_encoder)], with_synth=False)
    em = ctx.cfg.external.ddr_entropy_models
    folds = [f for f in em.folds if f in ctx.folds("deepdrid")] or ctx.folds("deepdrid")[:1]
    seeds = [s for s in em.seeds if s in ctx.seeds()] or ctx.seeds()[:1]
    P.stage_predict(ctx, "deepdrid", "entropy", eval_dataset="ddr", image_set="ddr_entropy", folds=folds, seeds=seeds)
    run_analysis(ctx, "exp4-ddr", lambda: run_ddr_analysis(ctx))


def exp4_ddr_repl(ctx: Ctx) -> None:
    from realcal import pipeline as P
    from realcal.experiments.exp4_external import run_ddr_repl_analysis

    P.stage_preprocess(ctx, "ddr", [224])
    P.stage_splits(ctx, "ddr")
    P.stage_features(ctx, "ddr", with_synth=False)
    P.stage_train(ctx, "ddr")
    P.stage_predict(ctx, "ddr", "real", eval_dataset="deepdrid", image_set="all")
    P.stage_predict(ctx, "ddr", "synth", eval_dataset="deepdrid", image_set="all")
    run_analysis(ctx, "exp4-ddr-repl", lambda: run_ddr_repl_analysis(ctx))


def sensitivity(ctx: Ctx, which: list[str]) -> None:
    from realcal import pipeline as P
    from realcal.experiments.sensitivity import ALL, run_sensitivity_analysis

    which = which or [w for w in ALL if w not in ("448", "aug")]
    sc = ctx.cfg.sensitivity
    if "448" in which:
        modes = list(sc.s448_modes) + (["ft"] if bool(sc.s448_include_ft) else [])
        P.stage_preprocess(ctx, "deepdrid", [224, 448])
        P.stage_corrupt(ctx, "deepdrid", [448])
        P.stage_features(ctx, "deepdrid", size=448, encoders=P.available_backbones(ctx))
        P.stage_train(ctx, "deepdrid", modes=modes, variant="s448")
        P.stage_predict(ctx, "deepdrid", "real", modes=modes, variant="s448")
        P.stage_predict(ctx, "deepdrid", "synth", modes=modes, variant="s448")
    if "aug" in which:
        bbs = [b for b in sc.aug_backbones if b in ctx.backbone_keys()] or ctx.backbone_keys()[:1]
        modes = list(sc.aug_modes)
        P.stage_train(ctx, "deepdrid", backbones=bbs, modes=modes, variant="aug")
        P.stage_predict(ctx, "deepdrid", "real", backbones=bbs, modes=modes, variant="aug")
        P.stage_predict(ctx, "deepdrid", "synth", backbones=bbs, modes=modes, variant="aug")
    run_analysis(ctx, "sensitivity-" + "-".join(which), lambda: run_sensitivity_analysis(ctx, which))


def report(ctx: Ctx, a) -> None:
    from realcal.report.figures import make_figures
    from realcal.report.fill_values import write_values
    from realcal.report.tables import make_tables

    every = not (a.tables or a.figures or a.values)
    if every or a.tables:
        make_tables(ctx)
    if every or a.figures:
        make_figures(ctx)
    if every or a.values:
        write_values(ctx)
    log.info("report written under %s", ctx.out)


def all_dev(ctx: Ctx, a) -> None:
    from realcal import pipeline as P
    from realcal.experiments.clinical import run_clinical
    from realcal.experiments.exp1_real import run_exp1
    from realcal.experiments.exp2_synth import run_exp2
    from realcal.experiments.exp3_calibrate import run_exp3
    from realcal.experiments.power import run_power

    ds = "deepdrid"
    t0 = time.time()
    P.stage_preprocess(ctx, ds)
    P.stage_splits(ctx, ds)
    P.stage_corrupt(ctx, ds)
    P.stage_features(ctx, ds)
    P.stage_train(ctx, ds)
    P.stage_predict(ctx, ds, "real")
    P.stage_predict(ctx, ds, "synth")
    for name, fn in [("exp1", lambda: run_exp1(ctx)), ("exp2", lambda: run_exp2(ctx)), ("exp3", lambda: run_exp3(ctx)),
                     ("clinical", lambda: run_clinical(ctx)), ("power", lambda: run_power(ctx))]:
        run_analysis(ctx, name, fn)
    sensitivity(ctx, ["sens", "perfold", "subgroup", "seedsd", "noise"])
    report(ctx, a)
    log.info("all-dev finished in %.1f h", (time.time() - t0) / 3600)


# ----------------------------------------------------------------------------- status

def status(ctx: Ctx) -> None:
    from realcal.scheduler import registry
    from realcal.utils.gpu import query_nvidia_smi

    for out_root, label in [(ctx.base_out, "MAIN"), (ctx.base_out / "smoke", "SMOKE")]:
        db = out_root / "jobs.sqlite"
        if not db.exists():
            continue
        c2 = Ctx(ctx.cfg, smoke=(label == "SMOKE"))
        reg = registry(c2)
        reg.reset_dead()
        counts = reg.counts()
        print(f"\n=== {label} ({out_root}) ===")
        print(f"{'stage':18s} {'done':>7s} {'running':>8s} {'failed':>7s} {'pending':>8s}")
        for st in sorted(counts):
            c = counts[st]
            print(f"{st:18s} {c.get('done', 0):7d} {c.get('running', 0):8d} {c.get('failed', 0):7d} {c.get('pending', 0):8d}")
        run = reg.rows("WHERE status='running'")
        if run:
            print("running:")
            for r in run[:40]:
                print(f"   {r[3] or '?':6s} pid {r[4]}  {r[0]}  ({(time.time() - (r[5] or time.time())) / 60:.1f} min)")
        failed = reg.rows("WHERE status='failed'")
        for r in failed[:10]:
            last = (r[8] or "").strip().splitlines()[-1:] or [""]
            print(f"   FAILED {r[0]}: {last[0][:160]}")
        reg.close()
    g = query_nvidia_smi()
    if g:
        print("\nGPU utilisation:")
        for x in g:
            print(f"   gpu{x.index} {x.name:28s} util {x.util:3d}%  free {x.free_mb:6d} / {x.total_mb} MB")
    logs = ctx.base_out / "logs"
    if logs.exists():
        latest: dict[str, os.DirEntry] = {}
        for e in os.scandir(logs):
            if e.name.endswith(".log"):
                st = e.name.rsplit("_", 2)[0]
                if st not in latest or e.stat().st_mtime > latest[st].stat().st_mtime:
                    latest[st] = e
        for st, e in sorted(latest.items()):
            print(f"\n--- {st}: {e.path}")
            with open(e.path, "rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                fh.seek(max(0, size - 4000))
                tail = fh.read().decode(errors="replace").splitlines()[-5:]
            for line in tail:
                print("   " + line[:200])


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    a = parse(argv)
    setup_logging()
    if a.stage != "fetch-weights":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    ctx = make_ctx(a)
    ctx.out.mkdir(parents=True, exist_ok=True)
    from realcal import pipeline as P

    st = a.stage
    if a.dataset == "all" and st != "eda":
        raise SystemExit("--dataset all is only valid for the eda stage")
    if st in ("features", "train", "predict-real", "predict-synth", "inspect", "preprocess", "splits", "corrupt-cache"):
        guard_external(ctx, a.dataset, st)
    if st == "eda":
        from realcal.data.eda import run_eda

        run_eda(ctx, ["deepdrid", "fives", "mshf", "ddr"] if a.dataset == "all" else [a.dataset])
    elif st == "inspect":
        from realcal.data.inspect import run_inspect

        run_inspect(ctx, a.dataset)
    elif st == "preprocess":
        P.stage_preprocess(ctx, a.dataset, a.sizes)
    elif st == "fetch-weights":
        from realcal.weights import fetch_all

        fetch_all(ctx)
    elif st == "splits":
        P.stage_splits(ctx, a.dataset)
    elif st == "corrupt-cache":
        P.stage_corrupt(ctx, a.dataset, a.sizes)
    elif st == "features":
        P.stage_features(ctx, a.dataset, size=a.size, encoders=a.backbones)
    elif st == "train":
        P.stage_train(ctx, a.dataset, **train_kw(a, ctx))
    elif st == "predict-real":
        P.stage_predict(ctx, a.dataset, "real", **train_kw(a, ctx))
    elif st == "predict-synth":
        P.stage_predict(ctx, a.dataset, "synth", **train_kw(a, ctx))
    elif st == "exp1":
        from realcal.experiments.exp1_real import run_exp1

        run_analysis(ctx, "exp1", lambda: run_exp1(ctx))
    elif st == "exp2":
        from realcal.experiments.exp2_synth import run_exp2

        run_analysis(ctx, "exp2", lambda: run_exp2(ctx))
    elif st == "exp3":
        from realcal.experiments.exp3_calibrate import run_exp3

        run_analysis(ctx, "exp3", lambda: run_exp3(ctx))
    elif st == "freeze":
        from realcal.prereg import freeze

        freeze(ctx, yes=a.yes)
    elif st == "exp4-fives":
        guard_external(ctx, "fives", st)
        exp4_fives(ctx)
    elif st == "exp4-mshf":
        guard_external(ctx, "mshf", st)
        exp4_mshf(ctx)
    elif st == "exp4-ddr":
        guard_external(ctx, "ddr", st)
        exp4_ddr(ctx)
    elif st == "exp4-ddr-repl":
        guard_external(ctx, "ddr", st)
        exp4_ddr_repl(ctx)
    elif st == "clinical":
        from realcal.experiments.clinical import run_clinical

        run_analysis(ctx, "clinical", lambda: run_clinical(ctx))
    elif st == "sensitivity":
        sensitivity(ctx, a.which)
    elif st == "power":
        from realcal.experiments.power import run_power

        run_analysis(ctx, "power", lambda: run_power(ctx))
    elif st == "report":
        report(ctx, a)
    elif st == "status":
        status(ctx)
    elif st == "all-dev":
        all_dev(ctx, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
