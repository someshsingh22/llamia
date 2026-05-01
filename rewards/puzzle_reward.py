"""VERL custom reward for the puzzle popularity+ELO task.

Function signature matches `verl.trainer.config.RewardModelConfig`'s
`custom_reward_function` contract (data_source, solution_str, ground_truth, extra_info).
"""
from __future__ import annotations

from typing import Any

try:
    from .parser import parse_popularity_elo
except ImportError:
    # When loaded via load_extern_object (standalone file, not package import)
    from rewards.parser import parse_popularity_elo

POP_SCALE = 50.0   # within ±50 popularity points → linear partial credit
ELO_SCALE = 400.0  # within ±400 Elo → linear partial credit


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, int],
    extra_info: dict[str, Any] | None = None,
) -> float:
    pop_pred, elo_pred = parse_popularity_elo(solution_str)
    if pop_pred is None or elo_pred is None:
        return 0.0  # format gate → multiplicative zero
    pop_err = abs(pop_pred - int(ground_truth["popularity"]))
    elo_err = abs(elo_pred - int(ground_truth["elo"]))
    r_pop = max(0.0, 1.0 - pop_err / POP_SCALE)
    r_elo = max(0.0, 1.0 - elo_err / ELO_SCALE)
    return 0.5 * r_pop + 0.5 * r_elo
