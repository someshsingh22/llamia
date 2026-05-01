"""Analyze puzzle_traces.jsonl — rewards, tool usage, format rate, errors.

Usage:
    python scripts/analyze_traces.py                         # default logs/puzzle_traces.jsonl
    python scripts/analyze_traces.py logs/puzzle_traces.jsonl
    python scripts/analyze_traces.py --last 200             # last N trajectories only
    python scripts/analyze_traces.py --examples 3          # show N low/high reward examples
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load(path: Path, last: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows[-last:] if last else rows


def stats(vals: list[float]) -> str:
    a = np.array(vals, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f}  [min {a.min():.4f}  max {a.max():.4f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="logs/puzzle_traces.jsonl")
    ap.add_argument("--last", type=int, default=None, help="Only use last N rows")
    ap.add_argument("--examples", type=int, default=2, help="Low/high reward examples to show")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No trace file at {path}", file=sys.stderr)
        sys.exit(1)

    rows = load(path, args.last)
    if not rows:
        print("Trace file is empty.")
        return

    n = len(rows)
    scores = [r["score"] for r in rows]
    fmt = [r["format_pass"] for r in rows]
    tool_counts = [r["num_tool_calls"] for r in rows]
    sol_lens = [r["solution_len"] for r in rows]
    pop_errs = [r["pop_err"] for r in rows if r.get("pop_err", -1) >= 0]
    elo_errs = [r["elo_err"] for r in rows if r.get("elo_err", -1) >= 0]

    # per-tool breakdown
    tool_counter: Counter = Counter()
    for r in rows:
        for t in r.get("tool_calls", []):
            tool_counter[t] += 1

    # per-step reward (group by ts bucket of 60s, rough proxy)
    print(f"=== Puzzle Reward Trace Analysis  ({n} trajectories) ===\n")
    print(f"Format pass rate : {100 * np.mean(fmt):.1f}%  ({int(np.sum(fmt))}/{n})")
    print(f"Reward           : {stats(scores)}")
    if pop_errs:
        print(f"Pop error (fmt✓) : {stats(pop_errs)}")
        print(f"Elo error (fmt✓) : {stats(elo_errs)}")
    print(f"Tool calls/traj  : {stats(tool_counts)}")
    print(f"Solution length  : {stats(sol_lens)}")

    print("\n--- Tool call distribution ---")
    total_calls = sum(tool_counter.values())
    for tool, cnt in tool_counter.most_common():
        print(f"  {tool:<20} {cnt:5d}  ({100*cnt/max(total_calls,1):.1f}%)")

    # reward histogram (5 buckets)
    buckets, edges = np.histogram(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.001])
    print("\n--- Reward histogram ---")
    for i, (lo, hi, cnt) in enumerate(zip(edges, edges[1:], buckets)):
        bar = "█" * int(40 * cnt / max(n, 1))
        print(f"  [{lo:.1f},{hi:.1f})  {cnt:4d}  {bar}")

    if args.examples > 0:
        sorted_rows = sorted(rows, key=lambda r: r["score"])
        print(f"\n--- {args.examples} lowest-reward trajectories ---")
        for r in sorted_rows[:args.examples]:
            _print_example(r)
        print(f"\n--- {args.examples} highest-reward trajectories ---")
        for r in sorted_rows[-args.examples:]:
            _print_example(r)


def _print_example(r: dict) -> None:
    print(f"  score={r['score']:.3f}  fmt={r['format_pass']}  "
          f"tools={r['tool_calls']}  "
          f"pop {r.get('pop_pred','?')}→{r.get('pop_true','?')}  "
          f"elo {r.get('elo_pred','?')}→{r.get('elo_true','?')}")
    if r.get("tail"):
        tail = r["tail"].replace("\n", " ")[-200:]
        print(f"    tail: {tail}")


if __name__ == "__main__":
    main()
