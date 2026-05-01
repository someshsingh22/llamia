"""Analyze puzzle_traces.jsonl — rewards, tool usage, format rate, errors.

Usage:
    python scripts/analyze_traces.py                         # default logs/puzzle_traces.jsonl
    python scripts/analyze_traces.py logs/puzzle_traces.jsonl
    python scripts/analyze_traces.py --split val             # only llamia-puzzle-val records
    python scripts/analyze_traces.py --last 1600             # last N trajectories
    python scripts/analyze_traces.py --examples 3            # show N low/high examples
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VAL_SOURCE = "llamia-puzzle-val"


def load(path: Path, last: int | None = None, split: str | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if split == "val" and r.get("data_source") != VAL_SOURCE:
                    continue
                if split == "train" and r.get("data_source") == VAL_SOURCE:
                    continue
                rows.append(r)
            except json.JSONDecodeError:
                pass
    return rows[-last:] if last else rows


def _stats(vals: list[float]) -> str:
    a = np.array(vals, dtype=float)
    return f"{a.mean():.4f} ± {a.std():.4f}  [min {a.min():.4f}  max {a.max():.4f}]"


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    ax, ay = np.array(x, dtype=float), np.array(y, dtype=float)
    rx = np.argsort(np.argsort(ax)).astype(float)
    ry = np.argsort(np.argsort(ay)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def _best_at_k(uid_scores: dict[str, list[float]], k: int) -> float:
    """Mean of (best score among k rollouts) over all prompts, via bootstrap."""
    rng = np.random.default_rng(42)
    vals = [np.array(v) for v in uid_scores.values() if len(v) >= 1]
    if not vals:
        return float("nan")
    results = []
    for v in vals:
        n = len(v)
        if n <= k:
            results.append(float(v.max()))
        else:
            samples = rng.choice(v, size=(200, k), replace=False)
            results.append(float(samples.max(axis=1).mean()))
    return float(np.mean(results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="logs/puzzle_traces.jsonl")
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None)
    ap.add_argument("--examples", type=int, default=2)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No trace file at {path}", file=sys.stderr)
        sys.exit(1)

    rows = load(path, args.last, args.split)
    if not rows:
        print("No records after filtering.")
        return

    n = len(rows)
    scores = [r["score"] for r in rows]
    fmts   = [r["format_pass"] for r in rows]
    tools  = [r["num_tool_calls"] for r in rows]
    lens   = [r["solution_len"] for r in rows]
    pop_errs = [r["pop_err"] for r in rows if r.get("pop_err", -1) >= 0]
    elo_errs = [r["elo_err"] for r in rows if r.get("elo_err", -1) >= 0]

    # Per-tool breakdown
    tool_counter: Counter = Counter()
    for r in rows:
        for t in r.get("tool_calls", []):
            tool_counter[t] += 1

    # Group by uid for rank correlation and best@k
    uid_scores:   dict[str, list[float]] = defaultdict(list)
    uid_pop_preds: dict[str, list[float]] = defaultdict(list)
    uid_elo_preds: dict[str, list[float]] = defaultdict(list)
    uid_pop_true:  dict[str, float] = {}
    uid_elo_true:  dict[str, float] = {}

    for r in rows:
        uid = r.get("uid") or f'{r.get("pop_true")}_{r.get("elo_true")}'
        uid_scores[uid].append(r["score"])
        if r["format_pass"] and r.get("pop_pred") is not None:
            uid_pop_preds[uid].append(float(r["pop_pred"]))
            uid_elo_preds[uid].append(float(r["elo_pred"]))
            uid_pop_true[uid] = float(r["pop_true"])
            uid_elo_true[uid] = float(r["elo_true"])

    # Rank correlation (mean prediction per prompt, format-passing only)
    mean_pop_pred, mean_elo_pred, true_pops, true_elos = [], [], [], []
    for uid in uid_pop_preds:
        mean_pop_pred.append(float(np.mean(uid_pop_preds[uid])))
        mean_elo_pred.append(float(np.mean(uid_elo_preds[uid])))
        true_pops.append(uid_pop_true[uid])
        true_elos.append(uid_elo_true[uid])

    pop_pred_errs = [abs(p - t) for p, t in zip(mean_pop_pred, true_pops)]
    elo_pred_errs = [abs(p - t) for p, t in zip(mean_elo_pred, true_elos)]

    # Best@k (all rollouts of each prompt)
    max_k = max(len(v) for v in uid_scores.values()) if uid_scores else 0

    # ── Print ────────────────────────────────────────────────────────────────
    split_tag = f"  [{args.split}]" if args.split else ""
    print(f"=== Puzzle Trace Analysis{split_tag}  ({n} trajectories, {len(uid_scores)} prompts) ===\n")
    print(f"Format pass rate : {100 * np.mean(fmts):.1f}%  ({int(np.sum(fmts))}/{n})")
    print(f"Reward           : {_stats(scores)}")
    print(f"Tool calls/traj  : {_stats(tools)}")
    print(f"Solution length  : {_stats(lens)}")

    if pop_errs:
        print(f"\n--- Conditional on format pass ({len(pop_errs)} trajectories) ---")
        print(f"Pop MAE  : {np.mean(pop_errs):.2f}  MSE: {np.mean(np.array(pop_errs)**2):.2f}")
        print(f"Elo MAE  : {np.mean(elo_errs):.2f}  MSE: {np.mean(np.array(elo_errs)**2):.2f}")

    if mean_pop_pred:
        print(f"\n--- Per-prompt (mean over format-passing rollouts, {len(mean_pop_pred)} prompts) ---")
        print(f"Pop MAE  : {np.mean(pop_pred_errs):.2f}  MSE: {np.mean(np.array(pop_pred_errs)**2):.2f}")
        print(f"Elo MAE  : {np.mean(elo_pred_errs):.2f}  MSE: {np.mean(np.array(elo_pred_errs)**2):.2f}")
        print(f"Pop Spearman ρ : {_spearman(mean_pop_pred, true_pops):.4f}")
        print(f"Elo Spearman ρ : {_spearman(mean_elo_pred, true_elos):.4f}")

    if uid_scores and max_k > 1:
        print(f"\n--- best@k (bootstrap, {len(uid_scores)} prompts, max {max_k} rollouts/prompt) ---")
        for k in range(1, max_k + 1):
            print(f"  best@{k}: {_best_at_k(uid_scores, k):.4f}")

    print("\n--- Tool call distribution ---")
    total_calls = sum(tool_counter.values())
    for tool, cnt in tool_counter.most_common():
        print(f"  {tool:<20} {cnt:5d}  ({100*cnt/max(total_calls,1):.1f}%)")

    buckets, edges = np.histogram(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.001])
    print("\n--- Reward histogram ---")
    for lo, hi, cnt in zip(edges, edges[1:], buckets):
        bar = "█" * int(40 * cnt / max(n, 1))
        print(f"  [{lo:.1f},{hi:.1f})  {cnt:4d}  {bar}")

    if args.examples > 0 and rows:
        sorted_rows = sorted(rows, key=lambda r: r["score"])
        print(f"\n--- {args.examples} lowest-reward trajectories ---")
        for r in sorted_rows[:args.examples]:
            _print_example(r)
        print(f"\n--- {args.examples} highest-reward trajectories ---")
        for r in sorted_rows[-args.examples:]:
            _print_example(r)


def _print_example(r: dict) -> None:
    print(f"  score={r['score']:.3f}  fmt={r['format_pass']}  uid={r.get('uid')}  "
          f"tools={r.get('tool_calls')}  "
          f"pop {r.get('pop_pred','?')}→{r.get('pop_true','?')}  "
          f"elo {r.get('elo_pred','?')}→{r.get('elo_true','?')}")
    if r.get("tail"):
        tail = r["tail"].replace("\n", " ")[-200:]
        print(f"    tail: {tail}")


if __name__ == "__main__":
    main()
