"""Export small example images (224 px, preprocessed) for the study-design figure.

Usage (from realcal_code, venv active):  python scripts/export_thumbs.py
Writes <out>/thumbs/<category>_<k>.png (4 candidates per category).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realcal.config import Ctx, load_config  # noqa: E402
from realcal.corruptions.cache import corrupt_path, eval_hq_ids  # noqa: E402
from realcal.corruptions.ops import run_op  # noqa: E402
from realcal.corruptions.registry import corruption_rng, main_conditions  # noqa: E402
from realcal.data.common import load_metadata  # noqa: E402
from realcal.data.preprocess import mask_path, preproc_path  # noqa: E402
from realcal.utils.io import load_gray, load_rgb  # noqa: E402

N, SIZE = 4, 224


def save(img: np.ndarray, p: Path) -> None:
    from PIL import Image

    Image.fromarray(img).save(p)


def main() -> None:
    ctx = Ctx(load_config())
    out = ctx.out / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    def export(ds: str, df, tag: str) -> None:
        ids = [i for i in df["image_id"].astype(str) if preproc_path(ctx, ds, SIZE, i).exists()]
        for k, i in enumerate(rng.choice(ids, size=min(N, len(ids)), replace=False) if ids else []):
            save(load_rgb(preproc_path(ctx, ds, SIZE, i)), out / f"{tag}_{k}.png")
        print(f"{tag}: {min(N, len(ids))} of {len(ids)}")

    for ds, groups in {
        "deepdrid": {"deepdrid_hq": lambda m: m[m["hq"].astype(bool)],
                     "deepdrid_lq": lambda m: m[m["lq_any"].astype(bool)]},
        "fives": {"fives_hq": lambda m: m[m["hq"].astype(bool)], "fives_lq": lambda m: m[m["lq_any"].astype(bool)]},
        "mshf": {"mshf_portable_lq": lambda m: m[m["lq_any"].astype(bool) & (m["camera"].astype(str) == "portable")],
                 "mshf_tabletop_hq": lambda m: m[m["hq"].astype(bool) & (m["camera"].astype(str) == "tabletop")]},
        "ddr": {"ddr_gradable": lambda m: m[~m["ungradable"].astype(bool)],
                "ddr_ungradable": lambda m: m[m["ungradable"].astype(bool)]},
    }.items():
        try:
            m = load_metadata(ctx, ds)
            for tag, f in groups.items():
                export(ds, f(m), tag)
        except Exception as e:  # noqa: BLE001
            print(f"{ds}: skipped ({e})")

    # synthetic corruptions of DeepDRiD HQ images, severity 4
    suite = ctx.cfg.corruption_suite
    conds = {c.corruption: c for c in main_conditions(suite) if c.severity == 4}
    ids = eval_hq_ids(ctx, "deepdrid")
    for k, iid in enumerate(rng.choice(ids, size=min(N, len(ids)), replace=False)):
        iid = str(iid)
        base = load_rgb(preproc_path(ctx, "deepdrid", SIZE, iid))
        save(base, out / f"synth_clean_{k}.png")
        mask = load_gray(mask_path(ctx, "deepdrid", SIZE, iid)) > 0
        for name in ("defocus", "uneven_illumination", "gaussian_noise", "colour_cast", "occlusion"):
            c = conds[name]
            p = corrupt_path(ctx, "deepdrid", SIZE, c.name, iid)
            img = load_rgb(p) if p.exists() else run_op(c.op, base, mask, c.param, corruption_rng(iid, c), c.spatial, suite)
            save(img, out / f"synth_{name}_{k}.png")
    print("thumbs ->", out)


if __name__ == "__main__":
    main()
