#!/usr/bin/env bash
# Launch GRPO+DAPO toy training for the puzzle popularity+ELO task.
# All 8 GPUs are shared with lc0 (3 GB/GPU is noise on 80 GB A100s).
# The external vllm at port 7000 must be STOPPED before running this — it
# holds ~73 GB/GPU and would OOM the FSDP training process.
set -euo pipefail
cd "$(dirname "$0")/.."

if pgrep -f "vllm serve Qwen" >/dev/null 2>&1; then
    echo "ERROR: external vllm is still running." >&2
    echo "  Stop it first:  pkill -f 'vllm serve Qwen'" >&2
    echo "  Wait for memory to free, then re-run." >&2
    exit 1
fi

source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "Starting GRPO toy run (puzzle popularity+ELO)…"
# configs/qwen3_puzzle_grpo.yaml is symlinked into verl's trainer/config/ so
# ppo_trainer (the defaults parent) resolves without --config-path override.
python -m verl.trainer.main_ppo \
    --config-name qwen3_puzzle_grpo \
    "$@"
