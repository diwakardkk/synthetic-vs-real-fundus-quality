# RealCal: do synthetic corruptions predict real-world failure of retinal AI?

The code for the paper's whole pipeline: training on high-quality (HQ) fundus images, the real low-quality (LQ) arm, the 75-condition synthetic arm, RealCal calibration (Levels 1a/1b/2), the pre-registration lock, external validation (FIVES, MSHF, DDR), clinical translation, sensitivity analyses, and the paper's tables, figures and `values.tex`.

Everything runs on the Linux GPU server. Every stage is **resumable**: re-run the same command and it continues where it stopped. Launch long stages with `scripts/launch.sh` so they keep running after you close the laptop.

## 1. Paths (edit once)

`configs/paths.yaml`:

```yaml
data_root: /path/to/data          # contains DeepDRiD/ FIVES/ MSHF/ DDR/
out_root: /path/to/data/realcal_runs
weights_root: /path/to/data/weights
```

Expected data folders (dataset-specific layouts are in `configs/datasets/*.yaml`):

```
NPJ/DeepDRiD/regular_fundus_images/{regular-fundus-training,regular-fundus-validation,Online-Challenge1&2-Evaluation}
NPJ/FIVES/<extracted rar>/{train,test}/Original/*.png  + "Quality Assessment.xlsx"
NPJ/MSHF/MSHF dataset 2.0/{Original/CFP, Original/Portable_camera, MSHF_quality_scores.xlsx}
NPJ/DDR/DDR-dataset/DR_grading/{train,valid,test}/ + {train,valid,test}.txt
```

## 2. Command sequence

```bash
# one-time
cd /path/to/data/realcal_code
conda env create -f environment.yml && conda activate realcal
# edit configs/paths.yaml
bash scripts/download_weights.sh

# check the data mapping first
python -m realcal.cli inspect --dataset deepdrid

# smoke test of the whole DeepDRiD chain (minutes)
bash scripts/launch.sh all-dev --smoke ; bash scripts/status.sh

# real DeepDRiD run (safe to close the laptop; re-run the same line to resume)
bash scripts/launch.sh all-dev
bash scripts/status.sh
python -m realcal.cli report

# later, after pre-registration stage 2
python -m realcal.cli freeze
bash scripts/launch.sh exp4-fives ; bash scripts/launch.sh exp4-mshf ; bash scripts/launch.sh exp4-ddr ; bash scripts/launch.sh exp4-ddr-repl
python -m realcal.cli report
```

Also available: `inspect` for `fives`, `mshf` and `ddr` (allowed before the lock); `sensitivity --which 448 aug` (GPU sensitivity runs); `power`; `clinical`; and single stages (`preprocess`, `splits`, `corrupt-cache`, `features`, `train`, `predict-real`, `predict-synth`, `exp1`, `exp2`, `exp3`).

Common flags: `--gpus 0,1,3`, `--jobs-per-gpu N`, `--smoke`, `--force` (redo the selected jobs), `--retry-failed` (requeue failed jobs, max 3 attempts), `--config-override key.sub=value` (repeatable). Training filters: `--backbones`, `--modes`, `--folds`, `--seeds`, `--variant main|s448|aug|cv`.

Stop a stage gracefully with `bash scripts/stop.sh all-dev`. Each worker finishes its current batch, saves `last.pt`, and marks its job pending.

## 3. What runs where

| Stage | GPU | Output |
|---|---|---|
| preprocess | CPU pool | `cache/preproc/<ds>/<size>/*.png` (+ FOV masks) |
| splits | CPU | `splits/<ds>_folds.csv`, `results/viability.json` |
| corrupt-cache | CPU pool | `cache/corrupt/<ds>/<size>/<condition>/*.png`, `figures/supp_corruption_examples.pdf` |
| features | GPU pool | `cache/features/<ds>/<encoder>/<size>/<condition>.npy` |
| train | GPU pool | `runs/train/...` (`best.pt`, `threshold.json`, `metrics.jsonl`, `config.yaml`, `provenance.json`) |
| predict-real/synth | GPU pool | `predictions/<ds>/<train job>/<condition>.parquet` |
| exp1, exp2, exp3, clinical, sensitivity, power | CPU (joblib) | `results/*.csv/json` |
| report | CPU | `tables/*.csv/.tex`, `figures/*.pdf/.png`, `results/values.tex`, `results/summary.md` |

The scheduler starts one worker per GPU slot. The number of slots comes from the per-job memory estimates in `configs/backbones.yaml` (`mem_mb`) and the free memory on each GPU, with a 15% margin. Override it with `--jobs-per-gpu`. A job that runs out of CUDA memory is requeued with half the batch size, and gradient accumulation keeps the effective batch at 32.

Smoke runs write to `out_root/smoke/` and never mix with real results.

## 4. Weights

`scripts/download_weights.sh` downloads everything it can into `weights_root`:

* **RETFound** (`YukunZhou/RETFound_mae_natureCFP`) and **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`) are gated. Accept their licences on Hugging Face, then run `huggingface-cli login` or `export HF_TOKEN=...`.
* **BiomedCLIP**, **DINOv2**, **ResNet-50**, **EfficientNet-B3** and **ConvNeXt-B** download automatically.
* **FLAIR** (`flair/flair_resnet.pth`) and **VisionFM** (`visionfm/VFM_Fundus_weights.pth`) are distributed through their GitHub READMEs (jusiro/FLAIR, ABILab-CUHK/VisionFM). Download them manually into the printed paths, or put the link in `url:` in `configs/backbones.yaml`.

A backbone with missing weights is reported and skipped. Its jobs do not block the others.

## 5. Pre-registration lock

`exp3` writes `prereg/frozen_weights.json`. `freeze` prints a summary and writes `prereg/stage2_lock.json` with its hash (interactive confirmation, or `--yes`). Any stage that trains, extracts features, predicts or analyses on FIVES, MSHF or DDR refuses to run until the lock exists and matches. `inspect`, `preprocess`, `splits` and `corrupt-cache` on those datasets are allowed before the lock.

## 6. Assumptions to verify on the server

See `ASSUMPTIONS.md`.
