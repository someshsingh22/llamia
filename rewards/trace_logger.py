"""Append-only JSONL trace logger + wandb metric flusher for puzzle rollouts.

JSONL writes are O_APPEND — atomic for lines < PIPE_BUF (4096 B) on Linux,
so multiple Ray worker processes can write concurrently without locking.

wandb flushing runs in the TaskRunner process (same process as wandb.init),
accumulating per-trajectory records and logging aggregate stats with
commit=False every PUZZLE_WANDB_FLUSH trajectories so VERL's own
wandb.log(step=N) commits everything in one atomic step update.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

_LOG_PATH = os.environ.get("PUZZLE_TRACE_LOG", "logs/puzzle_traces.jsonl")
# Flush wandb every N trajectories (default = full train batch size).
# Set to 0 to disable wandb flushing from the reward function.
_FLUSH_EVERY = int(os.environ.get("PUZZLE_WANDB_FLUSH", "64"))

# Module-level accumulator — safe because NaiveRewardManager iterates
# trajectories sequentially inside the single TaskRunner Ray actor.
_buf: list[dict] = []


def log_trace(record: dict) -> None:
    """Append record to JSONL and optionally flush aggregate stats to wandb."""
    # --- JSONL ---
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)

    # --- wandb accumulator ---
    if _FLUSH_EVERY <= 0:
        return
    _buf.append(record)
    if len(_buf) >= _FLUSH_EVERY:
        _flush_wandb()


def _flush_wandb() -> None:
    """Compute batch-level stats and log to the active wandb run (commit=False)."""
    if not _buf:
        return
    try:
        import wandb  # type: ignore
        if wandb.run is None:
            return

        scores = [r["score"] for r in _buf]
        fmts = [r["format_pass"] for r in _buf]
        tools = [r["num_tool_calls"] for r in _buf]
        lens = [r["solution_len"] for r in _buf]
        pop_errs = [r["pop_err"] for r in _buf if r.get("pop_err", -1) >= 0]
        elo_errs = [r["elo_err"] for r in _buf if r.get("elo_err", -1) >= 0]

        metrics: dict[str, Any] = {
            "reward/std":               float(np.std(scores)),
            "reward/format_pass_rate":  float(np.mean(fmts)),
            "reward/num_tool_calls":    float(np.mean(tools)),
            "reward/solution_len":      float(np.mean(lens)),
        }
        if pop_errs:
            metrics["reward/pop_err_mean"] = float(np.mean(pop_errs))
            metrics["reward/elo_err_mean"] = float(np.mean(elo_errs))

        wandb.log(metrics, commit=False)
    except Exception:
        pass  # never let logging break training
    finally:
        _buf.clear()
