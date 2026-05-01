"""VERL custom reward for the puzzle popularity+ELO task.

Function signature matches `verl.trainer.config.RewardModelConfig`'s
`custom_reward_function` contract (data_source, solution_str, ground_truth, extra_info).

Returns a dict so VERL stores per-trajectory metrics in batch.non_tensor_batch:
  score, format_pass, pop_err, elo_err, num_tool_calls, solution_len
Full structured traces are written to logs/puzzle_traces.jsonl (O_APPEND).
"""
from __future__ import annotations

import re
import time
from typing import Any

try:
    from .parser import parse_popularity_elo
    from .trace_logger import log_trace
except ImportError:
    # When loaded via load_extern_object (standalone file, not package import)
    from rewards.parser import parse_popularity_elo
    from rewards.trace_logger import log_trace

POP_SCALE = 50.0   # within ±50 popularity points → linear partial credit
ELO_SCALE = 400.0  # within ±400 Elo → linear partial credit

# Matches <tool_call>\n{"name": "foo", ...} in hermes format
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
        pop_err = elo_err = -1.0          # -1 sentinel used only in trace log
        verl_pop_err = POP_SCALE          # max-error fill for VERL val means
        verl_elo_err = ELO_SCALE
    else:
        pop_err = float(abs(pop_pred - int(ground_truth["popularity"])))
        elo_err = float(abs(elo_pred - int(ground_truth["elo"])))
        verl_pop_err = pop_err
        verl_elo_err = elo_err
        r_pop = max(0.0, 1.0 - pop_err / POP_SCALE)
        r_elo = max(0.0, 1.0 - elo_err / ELO_SCALE)
        score = 0.5 * r_pop + 0.5 * r_elo

    tool_calls = _TOOL_NAME_RE.findall(solution_str)
    num_tool_calls = len(tool_calls)

    try:
        log_trace({
            "ts": time.time(),
            "data_source": data_source,
            "score": score,
            "format_pass": format_pass,
            "pop_pred": pop_pred,
            "elo_pred": elo_pred,
            "pop_true": int(ground_truth.get("popularity", -1)),
            "elo_true": int(ground_truth.get("elo", -1)),
            "pop_err": pop_err,        # -1 for format failures (trace only)
            "elo_err": elo_err,
            "tool_calls": tool_calls,
            "num_tool_calls": num_tool_calls,
            "solution_len": len(solution_str),
            "num_turns": (extra_info or {}).get("num_turns"),
            "tail": solution_str[-400:],
        })
    except Exception:
        pass  # never let logging break training

    # verl_pop_err / verl_elo_err use POP_SCALE/ELO_SCALE for format failures
    # so VERL's process_validation_metrics mean@N is meaningful (not distorted by -1).
    return {
        "score": score,
        "format_pass": float(format_pass),
        "pop_err": verl_pop_err,
        "elo_err": verl_elo_err,
        "num_tool_calls": float(num_tool_calls),
        "solution_len": float(len(solution_str)),
    }
