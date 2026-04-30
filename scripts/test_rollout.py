"""Run N puzzles through ChessAnalyst, parse the final answer, score with
puzzle reward, and print aggregate stats.

Fail-fast diagnostic — NOT a unit test. Output is human-readable.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq

from agents.chess_analyst import ChessAnalyst
from rewards.parser import parse_popularity_elo
from rewards.puzzle_reward import compute_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/puzzle_val.parquet"))
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--llm-url", default="http://localhost:7000/v1")
    ap.add_argument("--lc0-url", default="http://localhost:7100")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()[: args.n]
    print(f"Running {len(rows)} puzzles…\n")

    scores: list[float] = []
    fmt_ok = 0
    tool_calls_per_run: list[int] = []
    rounds_per_run: list[int] = []
    t0 = time.time()

    for i, row in enumerate(rows):
        fen = row["fen"]
        gt = row["reward_model"]["ground_truth"]
        analyst = ChessAnalyst(
            llm_base_url=args.llm_url,
            model=args.model,
            lc0_url=args.lc0_url,
            initial_fen=fen,
        )
        # Use the user prompt from the dataset row directly.
        user_content = row["prompt"][-1]["content"]
        try:
            answer = analyst.run(user_content, verbose=False)
        except Exception as e:
            print(f"[{i:3d}] ROLLOUT ERROR: {type(e).__name__}: {e}")
            scores.append(0.0)
            continue

        pop, elo = parse_popularity_elo(answer)
        score = compute_score(
            data_source="x",
            solution_str=answer,
            ground_truth=gt,
            extra_info=None,
        )
        scores.append(score)
        if pop is not None:
            fmt_ok += 1
        stats = analyst.last_stats
        tool_calls_per_run.append(stats.get("tool_calls", 0))
        rounds_per_run.append(stats.get("llm_rounds", 0))

        print(
            f"[{i:3d}] gt=(pop={gt['popularity']}, elo={gt['elo']}) "
            f"pred=(pop={pop}, elo={elo}) score={score:.3f} "
            f"tools={stats.get('tool_calls')} rounds={stats.get('llm_rounds')}"
        )

    elapsed = time.time() - t0
    print("\n=== Rollout smoke-test summary ===")
    print(f"  N                : {len(scores)}")
    print(f"  Wall time        : {elapsed:.1f}s ({elapsed/max(1,len(scores)):.1f}s/puzzle)")
    print(f"  Format-pass rate : {fmt_ok / max(1, len(scores)):.1%}")
    print(f"  Mean reward      : {statistics.mean(scores):.3f}")
    print(f"  Reward stdev     : {statistics.pstdev(scores):.3f}")
    if tool_calls_per_run:
        print(
            f"  Tool calls       : mean={statistics.mean(tool_calls_per_run):.1f} "
            f"max={max(tool_calls_per_run)}"
        )
        print(
            f"  LLM rounds       : mean={statistics.mean(rounds_per_run):.1f} "
            f"max={max(rounds_per_run)}"
        )


if __name__ == "__main__":
    main()
