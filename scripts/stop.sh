#!/usr/bin/env bash
# Graceful stop: SIGTERM -> workers finish the current batch, save last.pt, mark jobs pending.
#   bash scripts/stop.sh <stage> [--smoke]
set -uo pipefail
if [ $# -lt 1 ]; then echo "usage: bash scripts/stop.sh <stage> [--smoke]"; exit 1; fi
STAGE="$1"; SUFFIX=""; [ "${2:-}" = "--smoke" ] && SUFFIX="_smoke"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$(command -v python || true)"
VENV="$ROOT/../realcal-env/bin/python"
if [ -x "$VENV" ] && ! "${PYTHON:-false}" -c "import omegaconf" 2>/dev/null; then PYTHON="$VENV"; fi
PYTHON="${PYTHON:-python3}"
OUT_ROOT="$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$ROOT/configs/paths.yaml'))['out_root'])")"
PIDFILE="$OUT_ROOT/pids/${STAGE}${SUFFIX}.pid"
if [ ! -f "$PIDFILE" ]; then echo "no pid file for $STAGE$SUFFIX"; exit 1; fi
PID="$(cat "$PIDFILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM "$PID"
  echo "SIGTERM sent to $STAGE$SUFFIX (pid $PID). Waiting up to 10 min for a clean exit..."
  for i in $(seq 1 600); do kill -0 "$PID" 2>/dev/null || { echo "stopped."; exit 0; }; sleep 1; done
  echo "still running; check with scripts/status.sh (send 'kill -9 $PID' only as a last resort)"
else
  echo "$STAGE$SUFFIX is not running"
fi
