#!/usr/bin/env bash
# Start 8 lc0 HTTP inference servers, one per GPU, on ports 7100–7107.
# Logs go to /tmp/lc0_server_<gpu>.log by default (override LOG_DIR).
#
# Usage:
#   ./lc0_server/scripts/start_lc0_servers.sh          # all 8 GPUs
#   N_GPUS=4 ./lc0_server/scripts/start_lc0_servers.sh # first 4 only
#   LOG_DIR=/var/log ./lc0_server/scripts/start_lc0_servers.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="$ROOT/.venv"
START_SCRIPT="$ROOT/scripts/start_lc0_server.sh"

N_GPUS="${N_GPUS:-8}"
BASE_PORT="${BASE_PORT:-7100}"
LOG_DIR="${LOG_DIR:-/tmp}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: lc0_server venv missing. Run: cd lc0_server && uv sync" >&2
    exit 1
fi

PIDS=()
for i in $(seq 0 $((N_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="$LOG_DIR/lc0_server_gpu${i}.log"
    echo "Starting lc0 server on GPU $i → port $PORT (log: $LOG)"
    CUDA_VISIBLE_DEVICES=$i LC0_PORT=$PORT \
        nohup bash "$START_SCRIPT" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Started ${#PIDS[@]} lc0 servers. Waiting for health checks..."
sleep 3

ALL_OK=1
for i in $(seq 0 $((N_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  GPU $i port $PORT — OK"
    else
        echo "  GPU $i port $PORT — NOT READY (check $LOG_DIR/lc0_server_gpu${i}.log)"
        ALL_OK=0
    fi
done

if [[ $ALL_OK -eq 1 ]]; then
    echo ""
    echo "All lc0 servers healthy. Ports: ${BASE_PORT}–$((BASE_PORT + N_GPUS - 1))"
else
    echo ""
    echo "Some servers failed — check logs above."
    exit 1
fi
