"""VERL custom reward for the puzzle popularity+ELO task.

Returns a dict so VERL stores per-trajectory metrics in batch.non_tensor_batch,
which flow through process_validation_metrics to wandb:
  score, format_pass, pop_err, elo_err, pop_sq_err, elo_sq_err,
  num_tool_calls, solution_len
Full structured traces (with uid, pop_pred, elo_pred) go to logs/puzzle_traces.jsonl.

Val metrics auto-logged by VERL at each val_freq:
  val/llamia-puzzle-val/score/mean@N .. best@1..8/mean (bootstrap)
  val/llamia-puzzle-val/pop_err/mean@N    (= MAE)
  val/llamia-puzzle-val/pop_sq_err/mean@N (= MSE)
  val/llamia-puzzle-val/elo_err/mean@N    (= MAE)
  val/llamia-puzzle-val/elo_sq_err/mean@N (= MSE)
  val/llamia-puzzle-val/format_pass/mean@N
  val/llamia-puzzle-val/num_tool_calls/mean@N

Rank correlation (Spearman's ρ) is computed from val records grouped by uid
in trace_logger._flush_val_wandb() and logged with commit=False.
"""
from __future__ import annotations

import re
import time
from typing import Any

try:
    from .parser import parse_popularity_elo
    from .trace_logger import log_trace
except ImportError:
    from rewards.parser import parse_popularity_elo
    from rewards.trace_logger import log_trace

POP_SCALE = 50.0   # within ±50 popularity → linear partial credit
ELO_SCALE = 400.0  # within ±400 Elo → linear partial credit

_TOOL_NAME_RE = re.compile(r"<tool_call>\s*\{[^}]*\"name\"\s*:\s*\"(\w+)\"", re.DOTALL)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, int],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pop_pred, elo_pred = parse_popularity_elo(solution_str)
    format_pass = int(pop_pred is not None and elo_pred is not None)

    if not format_pass:
        score = 0.0
        pop_err = elo_err = -1.0        # -1 sentinel in trace only
        verl_pop_err  = POP_SCALE       # max-error fill keeps VERL means meaningful
        verl_elo_err  = ELO_SCALE
        verl_pop_sq   = POP_SCALE ** 2
        verl_elo_sq   = ELO_SCALE ** 2
    else:
        pop_err = float(abs(pop_pred - int(ground_truth["popularity"])))
        elo_err = float(abs(elo_pred - int(ground_truth["elo"])))
        verl_pop_err  = pop_err
        verl_elo_err  = elo_err
        verl_pop_sq   = pop_err ** 2
        verl_elo_sq   = elo_err ** 2
        r_pop = max(0.0, 1.0 - pop_err / POP_SCALE)
        r_elo = max(0.0, 1.0 - elo_err / ELO_SCALE)
        score = 0.5 * r_pop + 0.5 * r_elo

    tool_calls = _TOOL_NAME_RE.findall(solution_str)
    ei = extra_info or {}

    try:
        log_trace({
            "ts":            time.time(),
            "data_source":   data_source,
            "uid":           ei.get("uid"),
            "fen":           ei.get("fen"),
            "score":         score,
            "format_pass":   format_pass,
            "pop_pred":      pop_pred,   # None for format failures
            "elo_pred":      elo_pred,
            "pop_true":      int(ground_truth.get("popularity", -1)),
            "elo_true":      int(ground_truth.get("elo", -1)),
            "pop_err":       pop_err,    # -1 sentinel for format failures (trace only)
            "elo_err":       elo_err,
            "tool_calls":    tool_calls,
            "num_tool_calls": len(tool_calls),
            "solution_len":  len(solution_str),
            "num_turns":     ei.get("num_turns"),
            "tail":          solution_str[-400:],
        })
    except Exception:
        pass

    return {
        "score":          score,
        "format_pass":    float(format_pass),
        "pop_err":        verl_pop_err,   # MAE component; POP_SCALE fill for failures
        "elo_err":        verl_elo_err,
        "pop_sq_err":     verl_pop_sq,    # MSE component
        "elo_sq_err":     verl_elo_sq,
        "num_tool_calls": float(len(tool_calls)),
        "solution_len":   float(len(solution_str)),
    }
