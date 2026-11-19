"""Detailed per-dataset tables (overview, splits, quality strata, grades, cameras/sites).

Usage (from realcal_code, venv active):
    python scripts/dataset_table.py            # uses cached metadata, builds it if missing
    python scripts/dataset_table.py --rebuild  # re-reads every dataset from disk

Prints Markdown tables and writes them to <out>/eda/dataset_tables.md. A dataset whose files are
missing or not yet extracted is listed with the error instead of stopping the script.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realcal.config import Ctx, load_config  # noqa: E402
from realcal.data.common import load_metadata  # noqa: E402

DATASETS = ["deepdrid", "fives", "mshf", "ddr"]
ROLE = {"deepdrid": "development", "fives": "external (4-class disease)", "mshf": "external (quality only)",
        "ddr": "external (DR grading)"}
TASK = {"deepdrid": "referable DR (grade >= 2)", "fives": "A/D/G/N (label 0-3)", "mshf": "none (quality labels)",
        "ddr": "referable DR (grade >= 2)"}


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
    line = lambda xs: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(xs)) + " |"  # noqa: E731
    return "\n".join([line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def pct(a: int, b: int) -> str:
    return f"{a} ({100 * a / b:.0f}%)" if b else "0"


def nuniq(s: pd.Series) -> int | str:
    s = s.dropna()
    return int(s.nunique()) if len(s) else "--"


def overview(name: str, m: pd.DataFrame) -> dict:
    n = len(m)
    lab = m["label"]
    bin_task = name in ("deepdrid", "ddr")
    return {"Dataset": name, "Role": ROLE[name], "Task": TASK[name], "Images": n,
            "Patients": nuniq(m["patient_id"]), "Eyes": nuniq(m["eye_id"]),
            "HQ": pct(int(m["hq"].sum()), n), "LQ": pct(int(m["lq_any"].sum()), n),
            "Ungradable": int(m["ungradable"].sum()),
            "Referable": pct(int((lab == 1).sum()), int(lab.notna().sum())) if bin_task else "--",
            "No label": int(lab.isna().sum()),
            "Splits": ", ".join(f"{k} {v}" for k, v in m["official_split"].value_counts().sort_index().items())}


def by_split(name: str, m: pd.DataFrame) -> list[dict]:
    out = []
    for sp, g in m.groupby(m["official_split"].fillna("?")):
        lab = g["label"]
        out.append({"Dataset": name, "Split": sp, "Images": len(g), "Patients": nuniq(g["patient_id"]),
                    "HQ": int(g["hq"].sum()), "LQ": int(g["lq_any"].sum()), "Ungradable": int(g["ungradable"].sum()),
                    "Referable": int((lab == 1).sum()) if name in ("deepdrid", "ddr") else "--",
                    "LQ & referable": int(((lab == 1) & g["lq_any"]).sum()) if name in ("deepdrid", "ddr") else "--"})
    return out


def strata(name: str, m: pd.DataFrame) -> list[dict]:
    out = []
    lq = m[m["lq_any"]]
    other = lq[~(lq["lq_artifact"] | lq["lq_clarity"] | lq["lq_field"])]
    for s, g in [("HQ", m[m["hq"]]), ("LQ any", lq), ("LQ artifact", m[m["lq_artifact"]]),
                 ("LQ clarity", m[m["lq_clarity"]]), ("LQ field", m[m["lq_field"]]), ("LQ no specific defect", other)]:
        if not len(g) and s not in ("HQ", "LQ any"):
            continue
        lab = g["label"]
        r = {"Dataset": name, "Stratum": s, "Images": len(g), "Patients": nuniq(g["patient_id"])}
        if name in ("deepdrid", "ddr"):
            r.update({"Referable": int((lab == 1).sum()), "Non-referable": int((lab == 0).sum()),
                      "% referable": f"{100 * (lab == 1).mean():.1f}" if lab.notna().any() else "--"})
        elif name == "fives":
            r.update({"Referable": "--", "Non-referable": "--",
                      "% referable": "classes " + "/".join(str(int((lab == k).sum())) for k in range(4))})
        else:
            r.update({"Referable": "--", "Non-referable": "--", "% referable": "--"})
        out.append(r)
    return out


def grades(name: str, m: pd.DataFrame) -> dict | None:
    if m["grade"].notna().sum() == 0:
        return None
    vc = m["grade"].value_counts()
    r = {"Dataset": name}
    for k in range(6):
        r[str(k)] = int(vc.get(k, 0))
    return r


def groups(name: str, m: pd.DataFrame) -> list[dict]:
    out = []
    key = "camera" if m["camera"].notna().any() else ("site" if m["site"].notna().any() else None)
    if key is None:
        return out
    for (k, s), g in m.groupby([m[key].fillna("?"), m["site"].fillna("?")] if key == "camera" else [m[key].fillna("?"), m[key].fillna("?")]):
        out.append({"Dataset": name, "Camera": k if key == "camera" else "--", "Site/source": s, "Images": len(g),
                    "HQ": int(g["hq"].sum()), "LQ": int(g["lq_any"].sum())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read datasets from disk instead of the cache")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ctx = Ctx(load_config())
    ov, sp, st, gr, gp, errs = [], [], [], [], [], []
    for name in a.datasets.split(","):
        try:
            m = load_metadata(ctx, name, rebuild=a.rebuild)
        except Exception as e:  # noqa: BLE001
            errs.append({"Dataset": name, "Status": f"NOT AVAILABLE: {type(e).__name__}: {str(e)[:120]}"})
            continue
        ov.append(overview(name, m))
        sp += by_split(name, m)
        st += strata(name, m)
        g = grades(name, m)
        if g:
            gr.append(g)
        gp += groups(name, m)
    parts = ["# Datasets used by RealCal", ""]
    for title, rows in [("1. Overview", ov), ("2. Official splits", sp), ("3. Quality strata", st),
                        ("4. Grade distribution (DR 0-4, 5 = ungradable; FIVES = class 0-3)", gr),
                        ("5. Cameras / sites", gp), ("Not available", errs)]:
        if rows:
            parts += [f"## {title}", "", md(pd.DataFrame(rows)), ""]
    text = "\n".join(parts)
    print(text)
    out = ctx.out / "eda" / "dataset_tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
