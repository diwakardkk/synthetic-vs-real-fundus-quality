#!/usr/bin/env bash
# Launch a RealCal stage detached from the terminal (survives SSH disconnect).
#   bash scripts/launch.sh <stage> [args...]
# e.g. bash scripts/launch.sh all-dev --smoke
#      bash scripts/launch.sh train --dataset deepdrid --gpus 0,1
# Uses tmux when available (attach with: tmux attach -t realcal_<stage>), otherwise nohup + setsid.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/launch.sh <stage> [args...]"; exit 1
fi
STAGE="$1"; shift
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$(command -v python || true)"
VENV="$ROOT/../realcal-env/bin/python"
if [ -x "$VENV" ] && ! "${PYTHON:-false}" -c "import omegaconf" 2>/dev/null; then PYTHON="$VENV"; fi
PYTHON="${PYTHON:-python3}"
OUT_ROOT="$("$PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$ROOT/configs/paths.yaml'))['out_root'])")"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids"

SUFFIX=""
for a in "$@"; do [ "$a" = "--smoke" ] && SUFFIX="_smoke"; done
NAME="realcal_${STAGE}${SUFFIX}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT_ROOT/logs/${STAGE}${SUFFIX}_${TS}.log"
PIDFILE="$OUT_ROOT/pids/${STAGE}${SUFFIX}.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Stage '$STAGE$SUFFIX' is already running (pid $(cat "$PIDFILE")). Use scripts/status.sh or scripts/stop.sh $STAGE."
  exit 1
fi

ARGS=""
for a in "$@"; do ARGS="$ARGS $(printf '%q' "$a")"; done
CMD="cd $(printf '%q' "$ROOT") && export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg && echo \$\$ > $(printf '%q' "$PIDFILE") && exec $(printf '%q' "$PYTHON") -m realcal.cli $STAGE$ARGS >> $(printf '%q' "$LOG") 2>&1"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$NAME" 2>/dev/null; then tmux kill-session -t "$NAME"; fi
  tmux new-session -d -s "$NAME" "bash -c $(printf '%q' "$CMD")"
  echo "started in tmux session '$NAME'   (attach: tmux attach -t $NAME ; detach: Ctrl+B then D)"
else
  setsid nohup bash -c "$CMD" > /dev/null 2>&1 < /dev/null &
  echo "started with nohup/setsid"
fi
sleep 2
echo "log : $LOG"
echo "pid : $( [ -f "$PIDFILE" ] && cat "$PIDFILE" || echo pending)"
echo "follow the log with:  tail -f $LOG"
