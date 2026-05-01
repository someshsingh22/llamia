#!/usr/bin/env bash
# Launch GRPO+DAPO toy training for the puzzle popularity+ELO task.
# All 8 GPUs are shared with lc0 (3 GB/GPU is noise on 80 GB A100s).
# The external vllm at port 7000 must be STOPPED before running this — it
# holds ~73 GB/GPU and would OOM the FSDP training process.
#
# Checkpoint behaviour:
#   (default)              new run at checkpoints/puzzle_grpo_toy/<TIMESTAMP>/
#   --resume               resume from the most recent timestamped run dir
#   --continue <path>      resume from <path> (any directory)
# Any other args are forwarded to verl as Hydra overrides.
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

CKPT_BASE="checkpoints/puzzle_grpo_toy"
PUZZLE_N_TRAIN="${PUZZLE_N_TRAIN:-4000}"
PUZZLE_N_TEST="${PUZZLE_N_TEST:-200}"
PUZZLE_SAMPLE_SEED="${PUZZLE_SAMPLE_SEED:-42}"

resume_flag=0
continue_path=""
passthrough=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            resume_flag=1
            shift
            ;;
        --continue)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --continue requires a checkpoint path argument." >&2
                exit 1
            fi
            continue_path="$2"
            shift 2
            ;;
        *)
            passthrough+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$continue_path" && $resume_flag -eq 1 ]]; then
    echo "ERROR: pass either --resume OR --continue <path>, not both." >&2
    exit 1
fi

if [[ -n "$continue_path" ]]; then
    if [[ ! -d "$continue_path" ]]; then
        echo "ERROR: --continue path does not exist: $continue_path" >&2
        exit 1
    fi
    run_dir="$continue_path"
    resume_mode="auto"
    echo "Continuing from explicit checkpoint dir: $run_dir"
elif [[ $resume_flag -eq 1 ]]; then
    if [[ ! -d "$CKPT_BASE" ]]; then
        echo "ERROR: no prior runs found under $CKPT_BASE; cannot --resume." >&2
        exit 1
    fi
    run_dir="$(find "$CKPT_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr | head -n1 | awk '{print $2}')"
    if [[ -z "$run_dir" ]]; then
        echo "ERROR: $CKPT_BASE has no subdirectories to resume from." >&2
        exit 1
    fi
    resume_mode="auto"
    echo "Resuming from latest run dir: $run_dir"
else
    ts="$(date +%Y%m%d_%H%M%S)"
    run_dir="${CKPT_BASE}/${ts}"
    mkdir -p "$run_dir"
    resume_mode="disable"
    echo "Starting fresh run at: $run_dir"
fi

echo "Materializing processed HF dataset splits for VERL..."
echo "  train: n=${PUZZLE_N_TRAIN} seed=${PUZZLE_SAMPLE_SEED}"
python -m data.prepare_puzzles materialize \
    --split train \
    --n-samples "$PUZZLE_N_TRAIN" \
    --seed "$PUZZLE_SAMPLE_SEED" \
    --out .cache/llamia_verl_data/puzzle_popularity_elo/train.parquet
echo "  test:  n=${PUZZLE_N_TEST} seed=${PUZZLE_SAMPLE_SEED}"
python -m data.prepare_puzzles materialize \
    --split test \
    --n-samples "$PUZZLE_N_TEST" \
    --seed "$PUZZLE_SAMPLE_SEED" \
    --out .cache/llamia_verl_data/puzzle_popularity_elo/test.parquet

echo "Starting GRPO toy run (puzzle popularity+ELO)…"
# configs/qwen3_puzzle_grpo.yaml is symlinked into verl's trainer/config/ so
# ppo_trainer (the defaults parent) resolves without --config-path override.
python -m verl.trainer.main_ppo \
    --config-name qwen3_puzzle_grpo \
    "trainer.default_local_dir=${run_dir}" \
    "trainer.resume_mode=${resume_mode}" \
    "${passthrough[@]}"
