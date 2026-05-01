"""Build the puzzle popularity+ELO parquet for VERL training.

Filters `ssingh22/llamia-chess-data` (config `behavioural_cloning`) for
`class == "type_Puzzle"`, extracts the FEN and the popularity/ELO targets
from the third gpt turn, and writes a parquet whose schema matches what
the VERL data loader expects: a `prompt` column (list[dict]) and a
`reward_model` column with `style`, `ground_truth`, and a side-channel
`fen` for the rollout-time tools.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GOLD_RE = re.compile(r"popularity is (-?\d+) and the ELO is (\d+)\b")

SYSTEM_PROMPT = (
    "You are a chess puzzle rater. You MUST output a numeric estimate; refusing or "
    "saying 'I cannot determine' is forbidden.\n"
    "A Lichess puzzle position has been loaded. You must produce two integers:\n"
    "  - popularity: -100..100 (Lichess upvote score; 100 = excellent puzzle)\n"
    "  - ELO: ~400..3000 (puzzle difficulty rating)\n"
    "You have access to lc0 engine tools (state shared across calls): "
    "get_position, analyze(nodes, multipv, moves=[]), get_policy(nodes), "
    "make_move(move), undo_move. Use them to inform your estimate — heavy "
    "tactics with forced wins and a clear best move tend to be popular and "
    "high-rated; quiet positions tend to score lower.\n"
    "Process: (1) call analyze() and/or get_policy() to gauge tactical sharpness, "
    "(2) make your best numeric estimate even if uncertain.\n"
    "Your reply MUST end with EXACTLY this line and nothing after it:\n"
    "  The popularity is <int> and the ELO is <int>\n"
    "Always emit this final line — guess if you must."
)

USER_TEMPLATE = (
    "FEN: {fen}\n"
    "What would be the popularity and the ELO of this puzzle? "
    "Popularity is between -100 and 100; ELO is the Lichess puzzle rating."
)


def build_rows(raw: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for ex in raw:
        if ex.get("class") != "type_Puzzle":
            continue
        state = ex.get("state") or []
        if not state:
            continue
        fen = state[0]
        gold_text = ""
        for msg in ex.get("conversations", []):
            if msg.get("from") == "gpt" and "popularity" in msg.get("value", "").lower():
                gold_text = msg["value"]
                break
        m = GOLD_RE.search(gold_text)
        if not m:
            continue
        pop, elo = int(m.group(1)), int(m.group(2))
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(fen=fen)},
            ],
            "reward_model": {
                "style": "puzzle_popularity_elo",
                "ground_truth": {"popularity": pop, "elo": elo},
            },
            "fen": fen,
            # uid: stable per puzzle, used by VERL's process_validation_metrics
            # to group multiple rollouts of the same prompt (for @N statistics).
            "uid": hashlib.md5(fen.encode()).hexdigest()[:12],
            "data_source": "ssingh22/llamia-chess-data",  # overridden per split below
        })
    return rows


def _stream_split(split: str):
    from datasets import load_dataset
    ds = load_dataset("ssingh22/llamia-chess-data", "behavioural_cloning", split=split, streaming=True)
    yield from ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-limit", type=int, default=4000)
    ap.add_argument("--val-limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible shuffle before taking the first N rows")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Each HF split is collected fully, then shuffled with the fixed seed before
    # taking the first N rows — ensures identical puzzles across re-runs.
    for hf_split, limit, name, data_source in [
        ("train", args.train_limit, "puzzle_train.parquet", "llamia-puzzle-train"),
        ("test",  args.val_limit,   "puzzle_val.parquet",   "llamia-puzzle-val"),
    ]:
        rows: list[dict] = []
        for ex in _stream_split(hf_split):
            built = build_rows([ex])
            rows.extend(built)
        if len(rows) < limit:
            raise RuntimeError(
                f"{name}: only found {len(rows)} puzzle rows (needed {limit}). "
                "The split may have fewer type_Puzzle examples than expected."
            )
        # Reproducible shuffle then slice
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(rows))[:limit]
        rows = [rows[i] for i in idx]
        # Stamp the split-specific data_source (used as wandb metric namespace)
        for r in rows:
            r["data_source"] = data_source
        table = pa.Table.from_pylist(rows)
        out = args.out_dir / name
        pq.write_table(table, out)
        print(f"{name}: {len(rows)} puzzles (seed={args.seed}) → {out}")


if __name__ == "__main__":
    main()
