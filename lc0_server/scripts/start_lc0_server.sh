#!/usr/bin/env bash
# Start the lc0 HTTP inference server using its own isolated UV environment.
#
# GPU footprint is ~3 GB regardless of MinibatchSize (cuBLAS workspace
# dominates), so we tune for throughput:
#   MinibatchSize=128  → ~7300 nps on A100 (3× over MinibatchSize=8)
#   MaxPrefetch=32     → upstream default, no memory cost
# Override any of LC0_MINIBATCH / LC0_MAX_PREFETCH / LC0_NNCACHE to retune.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="$ROOT/.venv"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: lc0_server venv missing. Run: cd lc0_server && uv sync" >&2
    exit 1
fi

export LC0_BIN="${LC0_BIN:-/dev/shm/somesh/lc0_src/build/release/lc0}"
export LC0_NET="${LC0_NET:-$ROOT/weights/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz}"
export LC0_BACKEND="${LC0_BACKEND:-cuda-fp16}"
export LC0_THREADS="${LC0_THREADS:-2}"
export LC0_MINIBATCH="${LC0_MINIBATCH:-128}"
export LC0_MAX_PREFETCH="${LC0_MAX_PREFETCH:-32}"
export LC0_NNCACHE="${LC0_NNCACHE:-200000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

HOST="${LC0_HOST:-0.0.0.0}"
PORT="${LC0_PORT:-7100}"

# Run from repo root so `lc0_server.server.app` is importable as a package.
cd "$REPO_ROOT"
exec "$VENV/bin/python" -m uvicorn \
    lc0_server.server.app:app \
    --host "$HOST" --port "$PORT" \
    --log-level info \
    --no-access-log
