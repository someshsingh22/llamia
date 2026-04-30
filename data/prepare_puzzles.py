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
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

GOLD_RE = re.compile(r"popularity is (-?\d+) and the ELO is (\d+)")

SYSTEM_PROMPT = (
    "You are a chess expert with access to lc0, a top neural network engine.\n"
    "A board has been loaded with the puzzle position. Use the engine tools to\n"
    "analyse it, then estimate two numbers:\n"
    "  - popularity: an integer in [-100, 100] (Lichess upvote score; 100 = excellent)\n"
    "  - ELO: an integer roughly in [400, 3000] (puzzle difficulty rating)\n"
    "Tools (state is shared across calls):\n"
    "  - get_position\n"
    "  - analyze(nodes, multipv, moves=[])\n"
    "  - get_policy(nodes)\n"
    "  - make_move(move) / undo_move\n"
    "When two tool calls are independent, emit BOTH in the same response — they run in parallel.\n"
    "When you are done analysing, finish with EXACTLY one line of the form:\n"
    "  The popularity is <int> and the ELO is <int>\n"
    "Do not output anything after that line."
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
            "data_source": "ssingh22/llamia-chess-data",
        })
    return rows


def _stream_split(split: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("ssingh22/llamia-chess-data", "behavioural_cloning", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-limit", type=int, default=4000)
    ap.add_argument("--val-limit", type=int, default=400)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, limit, name in [
        ("train", args.train_limit, "puzzle_train.parquet"),
        ("test", args.val_limit, "puzzle_val.parquet"),
    ]:
        rows = build_rows(_stream_split(split, limit * 5))  # 5x headroom for filter
        rows = rows[:limit]
        table = pa.Table.from_pylist(rows)
        out = args.out_dir / name
        pq.write_table(table, out)
        print(f"{name}: {len(rows)} puzzles → {out}")


if __name__ == "__main__":
    main()
