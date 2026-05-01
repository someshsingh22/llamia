"""Live reward monitor: tails puzzle_traces.jsonl and prints rolling stats.

Run alongside training:
    python scripts/watch_reward.py                    # default logs/puzzle_traces.jsonl
    python scripts/watch_reward.py --window 64       # rolling window size
    python scripts/watch_reward.py --interval 10     # refresh every 10 s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


def _fmt(vals):
    a = np.array(vals)
    return f"{a.mean():.3f}±{a.std():.3f} [{a.min():.3f},{a.max():.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="logs/puzzle_traces.jsonl")
    ap.add_argument("--window", type=int, default=64, help="Rolling window (trajectories)")
    ap.add_argument("--interval", type=float, default=5.0, help="Refresh interval (seconds)")
    args = ap.parse_args()

    path = Path(args.path)
    scores: deque = deque(maxlen=args.window)
    fmts: deque = deque(maxlen=args.window)
    tools: deque = deque(maxlen=args.window)
    lens: deque = deque(maxlen=args.window)
    total = 0

    print(f"Watching {path}  (window={args.window}, refresh={args.interval}s) — Ctrl-C to stop\n")

    file_pos = 0
    while True:
        if path.exists():
            with open(path) as f:
                f.seek(file_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    scores.append(r["score"])
                    fmts.append(r["format_pass"])
                    tools.append(r["num_tool_calls"])
                    lens.append(r["solution_len"])
                    total += 1
                file_pos = f.tell()

        if scores:
            fmt_rate = 100 * np.mean(list(fmts))
            print(
                f"\r[n={total:5d} | last {len(scores)}]  "
                f"reward {_fmt(list(scores))}  "
                f"fmt {fmt_rate:.0f}%  "
                f"tools {np.mean(list(tools)):.1f}  "
                f"len {np.mean(list(lens)):.0f}",
                end="",
                flush=True,
            )
        else:
            print(f"\r[waiting for traces at {path}]", end="", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
