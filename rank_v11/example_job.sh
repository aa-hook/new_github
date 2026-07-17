#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/challenge.jpg" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${RANK_V11_PORT:-8765}"
DEVICE="${RANK_V11_DEVICE:-auto}"
CACHE_DIR="${RANK_V11_CACHE_DIR:-${RUNNER_TEMP:-/tmp}/rank_v11_cache_${GITHUB_RUN_ID:-local}_${GITHUB_JOB:-job}}"
LOG_FILE="${RANK_V11_LOG:-${RUNNER_TEMP:-/tmp}/rank_v11_server.log}"

python "$ROOT/server.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --device "$DEVICE" \
  --mode accurate \
  --cache-dir "$CACHE_DIR" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

python "$ROOT/wait_ready.py" --url "http://127.0.0.1:${PORT}" --timeout 180 >/dev/null
python "$ROOT/solve.py" "$1" --url "http://127.0.0.1:${PORT}" --mode accurate
