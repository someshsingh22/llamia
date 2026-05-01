"""Append-only JSONL trace logger for puzzle rollouts.

Writes are O_APPEND — atomic for lines < PIPE_BUF (4096 B) on Linux,
so multiple Ray worker processes can write concurrently without locking.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_LOG_PATH = os.environ.get("PUZZLE_TRACE_LOG", "logs/puzzle_traces.jsonl")


def log_trace(record: dict) -> None:
    path = Path(_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)
