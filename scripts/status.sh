#!/usr/bin/env bash
# Show running stages, per-GPU utilisation, job counts per stage, and the last 5 log lines per stage.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$(command -v python || true)"
VENV="$ROOT/../realcal-env/bin/python"
if [ -x "$VENV" ] && ! "${PYTHON:-false}" -c "import omegaconf" 2>/dev/null; then PYTHON="$VENV"; fi
PYTHON="${PYTHON:-python3}"
OUT_ROOT="$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$ROOT/configs/paths.yaml'))['out_root'])")"

echo "=== running stages ==="
if [ -d "$OUT_ROOT/pids" ]; then
  for f in "$OUT_ROOT"/pids/*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"; st="$(basename "$f" .pid)"
    if kill -0 "$pid" 2>/dev/null; then echo "  RUNNING  $st (pid $pid)"; else echo "  stopped  $st"; fi
  done
fi
command -v tmux >/dev/null 2>&1 && tmux ls 2>/dev/null | grep realcal_ | sed 's/^/  tmux: /'
cd "$ROOT" && "$PYTHON" -m realcal.cli status 2>/dev/null
