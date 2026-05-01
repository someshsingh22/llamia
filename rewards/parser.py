"""Parse 'popularity is X and the ELO is Y' from rollout completions.

Tolerant: matches anywhere in the string; if multiple matches exist,
returns the LAST one (model often shows scratch work then concludes).
"""
from __future__ import annotations

import re

_PATTERN = re.compile(r"popularity\s+is\s+(-?\d+)\s+and\s+(?:the\s+)?ELO\s+is\s+(\d+)\b", re.IGNORECASE)


def parse_popularity_elo(text: str) -> tuple[int | None, int | None]:
    matches = list(_PATTERN.finditer(text or ""))
    if not matches:
        return (None, None)
    m = matches[-1]
    return int(m.group(1)), int(m.group(2))
