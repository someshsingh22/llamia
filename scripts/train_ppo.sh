#!/usr/bin/env bash
set -euo pipefail

VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${VLLM_PORT:-7000}"
N_GPUS="${N_GPUS:-8}"

source "$(dirname "$0")/../.venv/bin/activate"

python -m verl.trainer.main_ppo \
  --config-path "$(dirname "$0")/../configs" \
  --config-name qwen3_ppo_vllm \
  trainer.n_gpus_per_node="${N_GPUS}" \
  actor_rollout_ref.rollout.engine_kwargs.vllm.host="${VLLM_HOST}" \
  actor_rollout_ref.rollout.engine_kwargs.vllm.port="${VLLM_PORT}" \
  "$@"
