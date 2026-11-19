#!/usr/bin/env bash
# Download all pretrained weights into weights_root (configs/paths.yaml). Needs internet once.
# Gated models (RETFound, DINOv3): request access on their Hugging Face pages, then either
#   huggingface-cli login        or        export HF_TOKEN=hf_xxx
# FLAIR and VisionFM weights are on GitHub/Google Drive: if they are reported as MANUAL, download
# them and put them at the printed path (or set `url:` in configs/backbones.yaml and re-run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m realcal.cli fetch-weights "$@"
