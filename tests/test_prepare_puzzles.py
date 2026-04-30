"""Smoke-test the puzzle parquet builder on a tiny offline fixture."""
from pathlib import Path

import pyarrow.parquet as pq

from data.prepare_puzzles import build_rows, GOLD_RE


def test_gold_regex_extracts_popularity_and_elo():
    pop, elo = GOLD_RE.search("The popularity is 88 and the ELO is 1904").groups()
    assert int(pop) == 88
    assert int(elo) == 1904

    pop, elo = GOLD_RE.search("The popularity is -27 and the ELO is 2350").groups()
    assert int(pop) == -27
    assert int(elo) == 2350


def test_build_rows_extracts_fen_and_targets():
    raw = [{
        "class": "type_Puzzle",
        "state": ["6k1/5pp1/7p/8/1p4Pn/1N2P3/1n2KP1P/r2N3R b - - 1 34"],
        "conversations": [
            {"from": "human", "value": "ignored q1 <state>"},
            {"from": "gpt",   "value": "ignored a1"},
            {"from": "human", "value": "ignored q2"},
            {"from": "gpt",   "value": "ignored a2"},
            {"from": "human", "value": "What would be the popularity and the ELO of the puzzle..."},
            {"from": "gpt",   "value": "The popularity is 88 and the ELO is 1904"},
        ],
    }]

    rows = build_rows(raw)
    assert len(rows) == 1
    r = rows[0]
    assert r["fen"] == "6k1/5pp1/7p/8/1p4Pn/1N2P3/1n2KP1P/r2N3R b - - 1 34"
    assert r["reward_model"]["ground_truth"] == {"popularity": 88, "elo": 1904}
    assert isinstance(r["prompt"], list) and r["prompt"][0]["role"] == "system"
    assert "popularity" in r["prompt"][-1]["content"].lower()
    assert r["fen"] in r["prompt"][-1]["content"]


def test_non_puzzle_rows_are_dropped():
    raw = [{"class": "type_CCRL", "state": ["x"], "conversations": []}]
    assert build_rows(raw) == []
