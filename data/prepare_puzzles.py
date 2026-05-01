"""Publish and materialize the puzzle popularity+ELO VERL dataset.

The canonical processed dataset lives on Hugging Face:

    ssingh22/llamia-verl-data / puzzle_popularity_elo

with splits ``train`` and ``test``.  The processed ``train`` split contains all
raw train puzzles, while ``test`` is a fixed seeded sample from raw test.  This
module can rebuild and publish that dataset from ``ssingh22/llamia-chess-data``
and can materialize sampled parquet files when VERL requires local paths.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATASET_ID = "ssingh22/llamia-chess-data"
RAW_CONFIG_NAME = "behavioural_cloning"
PROCESSED_DATASET_ID = "ssingh22/llamia-verl-data"
PROCESSED_CONFIG_NAME = "puzzle_popularity_elo"
PROCESSED_CACHE_DIR = Path(".cache/llamia_verl_data") / PROCESSED_CONFIG_NAME
DEFAULT_TEST_SIZE = 1000
DEFAULT_SAMPLE_SEED = 42

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


def _data_source(split: str) -> str:
    return f"{PROCESSED_CONFIG_NAME}/{split}"


def build_rows(raw: Iterable[dict], split: str = "all") -> list[dict]:
    """Convert raw LLAMIA chess examples into VERL-ready puzzle rows.

    Args:
        raw: Raw examples from ``ssingh22/llamia-chess-data``.
        split: Processed Hugging Face split name used for the ``data_source``
            side channel.

    Returns:
        Rows with VERL ``prompt`` and ``reward_model`` fields plus ``fen``,
        stable ``uid``, and split-aware ``data_source``.
    """
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
            "data_source": _data_source(split),
        })
    return rows


def sample_rows(rows: list[dict], n_samples: int | None, seed: int) -> list[dict]:
    """Sample rows deterministically without modifying row contents.

    Args:
        rows: Candidate rows.
        n_samples: Number of rows to sample. ``None`` returns all rows.
        seed: RNG seed used when sampling.

    Returns:
        Deterministically sampled rows.
    """
    if n_samples is None:
        return list(rows)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if n_samples > len(rows):
        raise ValueError(f"requested {n_samples} rows, but only {len(rows)} are available")
    indices = random.Random(seed).sample(range(len(rows)), n_samples)
    return [rows[index] for index in indices]


def build_split_rows(
    raw_by_split: Mapping[str, Iterable[dict]],
    n_test: int = DEFAULT_TEST_SIZE,
    test_seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, list[dict]]:
    """Build train/test processed splits.

    Args:
        raw_by_split: Raw examples keyed by source split. The ``train`` and
            ``test`` keys are required.
        n_test: Number of test examples to publish.
        test_seed: Seed for selecting the fixed processed test set.

    Returns:
        A dict containing processed ``train`` and seeded ``test`` splits.
    """
    train_rows = build_rows(raw_by_split["train"], split="train")
    test_rows = build_rows(raw_by_split["test"], split="test")
    return {"train": train_rows, "test": sample_rows(test_rows, n_samples=n_test, seed=test_seed)}


def write_parquet_rows(rows: list[dict], out_path: Path) -> Path:
    """Write processed rows to a parquet file for VERL consumers.

    Args:
        rows: Processed puzzle rows.
        out_path: Destination parquet path.

    Returns:
        The destination path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return out_path


def load_raw_split(split: str, dataset_id: str = RAW_DATASET_ID, config_name: str = RAW_CONFIG_NAME):
    """Load one raw split from the upstream HF dataset."""
    from datasets import load_dataset

    load_dotenv(REPO_ROOT / ".env")
    return load_dataset(dataset_id, config_name, split=split)


def build_hf_dataset_dict(
    raw_dataset_id: str = RAW_DATASET_ID,
    raw_config_name: str = RAW_CONFIG_NAME,
    n_test: int = DEFAULT_TEST_SIZE,
    test_seed: int = DEFAULT_SAMPLE_SEED,
):
    """Build the processed Hugging Face ``DatasetDict``.

    Args:
        raw_dataset_id: Source dataset repository id.
        raw_config_name: Source dataset config/subset.
        n_test: Number of fixed test examples to publish.
        test_seed: Seed for selecting the fixed processed test set.

    Returns:
        A ``DatasetDict`` with ``train`` and ``test`` splits.
    """
    from datasets import Dataset, DatasetDict

    raw_by_split = {
        "train": load_raw_split("train", raw_dataset_id, raw_config_name),
        "test": load_raw_split("test", raw_dataset_id, raw_config_name),
    }
    split_rows = build_split_rows(raw_by_split, n_test=n_test, test_seed=test_seed)
    return DatasetDict({
        split: Dataset.from_list(rows)
        for split, rows in split_rows.items()
    })


def publish_processed_dataset(
    dataset_id: str = PROCESSED_DATASET_ID,
    config_name: str = PROCESSED_CONFIG_NAME,
    raw_dataset_id: str = RAW_DATASET_ID,
    raw_config_name: str = RAW_CONFIG_NAME,
    n_test: int = DEFAULT_TEST_SIZE,
    test_seed: int = DEFAULT_SAMPLE_SEED,
) -> None:
    """Publish the processed dataset to Hugging Face Hub.

    Args:
        dataset_id: Destination HF dataset repository id.
        config_name: Destination config/subset name.
        raw_dataset_id: Source HF dataset repository id.
        raw_config_name: Source HF config/subset name.
        n_test: Number of fixed test examples to publish.
        test_seed: Seed for selecting the fixed processed test set.
    """
    token = os.environ.get("HF_TOKEN")
    ds = build_hf_dataset_dict(
        raw_dataset_id=raw_dataset_id,
        raw_config_name=raw_config_name,
        n_test=n_test,
        test_seed=test_seed,
    )
    ds.push_to_hub(dataset_id, config_name=config_name, token=token)


def load_processed_split(
    split: str,
    dataset_id: str = PROCESSED_DATASET_ID,
    config_name: str = PROCESSED_CONFIG_NAME,
):
    """Load one processed split from the canonical HF dataset."""
    from datasets import load_dataset

    load_dotenv(REPO_ROOT / ".env")
    return load_dataset(dataset_id, config_name, split=split)


def materialize_processed_split(
    split: str,
    out_path: Path | None = None,
    dataset_id: str = PROCESSED_DATASET_ID,
    config_name: str = PROCESSED_CONFIG_NAME,
    n_samples: int | None = None,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Path:
    """Download one processed split and write it to parquet.

    Args:
        split: Processed HF split name.
        out_path: Optional destination. Defaults to the local cache path used by
            training scripts.
        dataset_id: Source processed HF dataset repository id.
        config_name: Source processed HF config/subset name.
        n_samples: Optional sampled row count for local materialization.
        seed: RNG seed used when sampling.

    Returns:
        The parquet file path.
    """
    rows = list(load_processed_split(split, dataset_id=dataset_id, config_name=config_name))
    rows = sample_rows(rows, n_samples=n_samples, seed=seed)
    target = out_path or (PROCESSED_CACHE_DIR / f"{split}.parquet")
    return write_parquet_rows(rows, target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish or materialize the processed puzzle VERL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    publish = subcommands.add_parser("publish", help="Build processed splits and push to Hugging Face")
    publish.add_argument("--dataset-id", default=PROCESSED_DATASET_ID)
    publish.add_argument("--config-name", default=PROCESSED_CONFIG_NAME)
    publish.add_argument("--raw-dataset-id", default=RAW_DATASET_ID)
    publish.add_argument("--raw-config-name", default=RAW_CONFIG_NAME)
    publish.add_argument("--n-test", type=int, default=DEFAULT_TEST_SIZE)
    publish.add_argument("--test-seed", type=int, default=DEFAULT_SAMPLE_SEED)

    materialize = subcommands.add_parser("materialize", help="Write a processed HF split to parquet")
    materialize.add_argument("--split", choices=["train", "test"], required=True)
    materialize.add_argument("--out", type=Path, default=None)
    materialize.add_argument("--dataset-id", default=PROCESSED_DATASET_ID)
    materialize.add_argument("--config-name", default=PROCESSED_CONFIG_NAME)
    materialize.add_argument("--n-samples", type=int, default=None)
    materialize.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the dataset publisher/materializer CLI."""
    args = _parse_args()
    if args.command == "publish":
        publish_processed_dataset(
            dataset_id=args.dataset_id,
            config_name=args.config_name,
            raw_dataset_id=args.raw_dataset_id,
            raw_config_name=args.raw_config_name,
            n_test=args.n_test,
            test_seed=args.test_seed,
        )
        print(f"Published {args.dataset_id}/{args.config_name}")
        return

    out_path = materialize_processed_split(
        split=args.split,
        out_path=args.out,
        dataset_id=args.dataset_id,
        config_name=args.config_name,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    print(f"Materialized {args.dataset_id}/{args.config_name}:{args.split} -> {out_path}")


if __name__ == "__main__":
    main()
