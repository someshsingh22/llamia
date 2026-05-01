"""Append-only JSONL trace logger + wandb metric flusher for puzzle rollouts.

JSONL writes are O_APPEND — atomic for lines < PIPE_BUF (4096 B) on Linux,
safe across Ray worker processes without locking.

Train buffer: flushes every PUZZLE_WANDB_FLUSH trajectories (default 64).
  Logs reward/std, format_pass_rate, num_tool_calls, pop_err_mean, elo_err_mean.

Val buffer: flushes every PUZZLE_VAL_FLUSH trajectories (default val_size × rollout_n
= 200 × 8 = 1600). After flushing, computes and logs:
  - Spearman's ρ for popularity and ELO predictions (format-passing only)
  - MAE and MSE (redundant with VERL's process_validation_metrics, logged here
    as a sanity check with the -1 sentinel correctly excluded)
All wandb calls use commit=False so they attach to VERL's own wandb.log(step=N).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_LOG_PATH   = os.environ.get("PUZZLE_TRACE_LOG",   "logs/puzzle_traces.jsonl")
_VAL_SOURCE = os.environ.get("PUZZLE_VAL_SOURCE",  "puzzle_popularity_elo/test")
_TRAIN_FLUSH = int(os.environ.get("PUZZLE_WANDB_FLUSH", "64"))
# 200 val prompts × 8 rollouts; override if val_size or rollout_n differ.
_VAL_FLUSH   = int(os.environ.get("PUZZLE_VAL_FLUSH", "1600"))

# Module-level accumulators — safe because NaiveRewardManager iterates
# trajectories sequentially inside the single TaskRunner Ray actor.
_train_buf: list[dict] = []
_val_buf:   list[dict] = []


def log_trace(record: dict) -> None:
    """Write record to JSONL and trigger flush if threshold reached."""
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)

    if record.get("data_source") == _VAL_SOURCE:
        _val_buf.append(record)
        if len(_val_buf) >= _VAL_FLUSH:
            _flush_val_wandb()
    else:
        _train_buf.append(record)
        if len(_train_buf) >= _TRAIN_FLUSH:
            _flush_train_wandb()


# ── Train flush ───────────────────────────────────────────────────────────────

def _flush_train_wandb() -> None:
    if not _train_buf:
        return
    try:
        import wandb  # type: ignore
        if wandb.run is None:
            return
        scores = [r["score"]         for r in _train_buf]
        fmts   = [r["format_pass"]   for r in _train_buf]
        tools  = [r["num_tool_calls"] for r in _train_buf]
        lens   = [r["solution_len"]  for r in _train_buf]
        pop_errs = [r["pop_err"] for r in _train_buf if r.get("pop_err", -1) >= 0]
        elo_errs = [r["elo_err"] for r in _train_buf if r.get("elo_err", -1) >= 0]
        m: dict[str, Any] = {
            "reward/std":              float(np.std(scores)),
            "reward/format_pass_rate": float(np.mean(fmts)),
            "reward/num_tool_calls":   float(np.mean(tools)),
            "reward/solution_len":     float(np.mean(lens)),
        }
        if pop_errs:
            m["reward/pop_err_mean"] = float(np.mean(pop_errs))
            m["reward/elo_err_mean"] = float(np.mean(elo_errs))
        wandb.log(m, commit=False)
    except Exception:
        pass
    finally:
        _train_buf.clear()


# ── Val flush ─────────────────────────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman's ρ via rank-transformed Pearson."""
    if len(x) < 3:
        return float("nan")
    ax, ay = np.array(x, dtype=float), np.array(y, dtype=float)
    rx = np.argsort(np.argsort(ax)).astype(float)
    ry = np.argsort(np.argsort(ay)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def _flush_val_wandb() -> None:
    """Aggregate val-set records and log rank correlation + summary stats."""
    if not _val_buf:
        return
    try:
        # Group format-passing rollouts by prompt uid.
        uid_pops:  dict[str, list[float]] = defaultdict(list)
        uid_elos:  dict[str, list[float]] = defaultdict(list)
        uid_pop_true: dict[str, float] = {}
        uid_elo_true: dict[str, float] = {}

        scores, fmts, tools = [], [], []
        for r in _val_buf:
            scores.append(r["score"])
            fmts.append(r["format_pass"])
            tools.append(r["num_tool_calls"])
            if r["format_pass"] and r.get("pop_pred") is not None:
                uid = r.get("uid") or f'{r["pop_true"]}_{r["elo_true"]}'
                uid_pops[uid].append(float(r["pop_pred"]))
                uid_elos[uid].append(float(r["elo_pred"]))
                uid_pop_true[uid] = float(r["pop_true"])
                uid_elo_true[uid] = float(r["elo_true"])

        # Mean prediction per prompt (over format-passing rollouts).
        mean_pop_pred, mean_elo_pred = [], []
        true_pops,     true_elos     = [], []
        for uid in uid_pops:
            mean_pop_pred.append(float(np.mean(uid_pops[uid])))
            mean_elo_pred.append(float(np.mean(uid_elos[uid])))
            true_pops.append(uid_pop_true[uid])
            true_elos.append(uid_elo_true[uid])

        # MAE / MSE (excluding format failures; -1 sentinel was for trace only)
        pop_errs = [abs(p - t) for p, t in zip(mean_pop_pred, true_pops)]
        elo_errs = [abs(p - t) for p, t in zip(mean_elo_pred, true_elos)]

        metrics: dict[str, Any] = {
            "val_custom/format_pass_rate":  float(np.mean(fmts)),
            "val_custom/reward_mean":        float(np.mean(scores)),
            "val_custom/reward_std":         float(np.std(scores)),
            "val_custom/num_tool_calls":     float(np.mean(tools)),
            "val_custom/n_prompts_fmt_pass": len(mean_pop_pred),
        }
        if mean_pop_pred:
            metrics["val_custom/pop_spearman_rho"] = _spearman(mean_pop_pred, true_pops)
            metrics["val_custom/elo_spearman_rho"] = _spearman(mean_elo_pred, true_elos)
            metrics["val_custom/pop_mae"]  = float(np.mean(pop_errs))
            metrics["val_custom/elo_mae"]  = float(np.mean(elo_errs))
            metrics["val_custom/pop_mse"]  = float(np.mean([e**2 for e in pop_errs]))
            metrics["val_custom/elo_mse"]  = float(np.mean([e**2 for e in elo_errs]))

        try:
            import wandb  # type: ignore
            if wandb.run is not None:
                wandb.log(metrics, commit=False)
        except Exception:
            pass

        # Always print to console as a backup.
        _print_val_summary(metrics, len(_val_buf))
    except Exception:
        pass
    finally:
        _val_buf.clear()


def _print_val_summary(m: dict, n_traj: int) -> None:
    print(
        f"\n[val_custom | {n_traj} trajectories]\n"
        f"  format_pass={m.get('val_custom/format_pass_rate',float('nan')):.2%}  "
        f"reward={m.get('val_custom/reward_mean',float('nan')):.3f}±{m.get('val_custom/reward_std',float('nan')):.3f}\n"
        f"  pop_spearman_ρ={m.get('val_custom/pop_spearman_rho',float('nan')):.3f}  "
        f"elo_spearman_ρ={m.get('val_custom/elo_spearman_rho',float('nan')):.3f}\n"
        f"  pop_MAE={m.get('val_custom/pop_mae',float('nan')):.1f}  "
        f"elo_MAE={m.get('val_custom/elo_mae',float('nan')):.1f}  "
        f"pop_MSE={m.get('val_custom/pop_mse',float('nan')):.1f}  "
        f"elo_MSE={m.get('val_custom/elo_mse',float('nan')):.1f}"
    )
