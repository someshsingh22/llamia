#!/usr/bin/env python
"""LLAMIA Chess Analyst CLI.

Usage:
    # vLLM (default)
    python scripts/run_analyst.py "Why is Nxd4 a blunder here, rnbqkb1r/pp2pppp/5n2/6B1/2pp4/4PN2/PP3PPP/RN1QKB1R w KQkq - 0 6"

    # Azure GPT-5 (reads AZURE_* from .env)
    python scripts/run_analyst.py "Best plan?" --provider azure --model gpt-5

    # Azure-hosted Claude via Anthropic native API
    python scripts/run_analyst.py "Best plan?" --provider azure --model claude-sonnet-4-6

    # Direct Anthropic API
    python scripts/run_analyst.py "Best plan?" --provider anthropic --model claude-opus-4-6
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from agents.chess_analyst import ChessAnalyst

_DEFAULT_MODELS = {
    "vllm": "Qwen/Qwen3-4B",
    "azure": "gpt-5",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


def main():
    p = argparse.ArgumentParser(description="LLAMIA Chess Analyst")
    p.add_argument("query", help="Analysis query — may contain an embedded FEN")
    p.add_argument("--provider", choices=["vllm", "azure", "anthropic", "openai"],
                   default="vllm", help="LLM provider (default: vllm)")
    p.add_argument("--model", default=None, help="Model/deployment name (provider default if omitted)")
    p.add_argument("--llm-url", default="http://localhost:7000/v1", help="Base URL for vLLM server")
    p.add_argument("--lc0-url", default="http://localhost:7100")
    p.add_argument("--fen", default=None, help="Explicit starting FEN")
    p.add_argument("--quiet", action="store_true", help="Only print final answer")
    args = p.parse_args()

    model = args.model or _DEFAULT_MODELS[args.provider]

    analyst = ChessAnalyst(
        llm_base_url=args.llm_url,
        model=model,
        lc0_url=args.lc0_url,
        initial_fen=args.fen,
        provider=args.provider,
    )

    if not args.quiet:
        try:
            h = analyst.lc0.health()
            print(f"[lc0]  {h['engine_id']['name']}  backend={h['backend']}")
        except Exception as e:
            print(f"[ERROR] lc0 unreachable at {args.lc0_url}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[LLM]  {model}  (provider={args.provider})\n")

    analyst.run(args.query, verbose=not args.quiet)


if __name__ == "__main__":
    main()
