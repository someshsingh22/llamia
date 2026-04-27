#!/usr/bin/env python
"""LLAMIA Chess Analyst CLI.

Usage:
    python scripts/run_analyst.py "Why is Nxd4 a blunder, rnbqkb1r/pp2pppp/5n2/6B1/2pp4/4PN2/PP3PPP/RN1QKB1R w KQkq - 0 6"
    python scripts/run_analyst.py "What is the best plan?" --fen "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
"""
from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.chess_analyst import ChessAnalyst


def main():
    p = argparse.ArgumentParser(description="LLAMIA Chess Analyst")
    p.add_argument("query", help="Analysis query — may contain an embedded FEN")
    p.add_argument("--llm-url", default="http://localhost:7000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8")
    p.add_argument("--lc0-url", default="http://localhost:7100")
    p.add_argument("--fen", default=None, help="Explicit starting FEN (overrides embedded FEN)")
    p.add_argument("--quiet", action="store_true", help="Only print final answer")
    args = p.parse_args()

    analyst = ChessAnalyst(
        llm_base_url=args.llm_url,
        model=args.model,
        lc0_url=args.lc0_url,
        initial_fen=args.fen,
    )

    # Health checks
    if not args.quiet:
        try:
            h = analyst.lc0.health()
            print(f"[lc0]  {h['engine_id']['name']}  backend={h['backend']}  weights={h['weights']}")
        except Exception as e:
            print(f"[ERROR] lc0 server unreachable at {args.lc0_url}: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            models = [m.id for m in analyst.client.models.list().data]
            print(f"[LLM]  {args.model}  ({args.llm_url})")
            print()
        except Exception as e:
            print(f"[ERROR] LLM server unreachable at {args.llm_url}: {e}", file=sys.stderr)
            sys.exit(1)

    analyst.run(args.query, verbose=not args.quiet)


if __name__ == "__main__":
    main()
