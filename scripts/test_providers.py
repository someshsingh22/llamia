#!/usr/bin/env python
"""Smoke-test ChessAnalyst across all configured providers.

Tests Azure Foundry deployments: gpt-5, claude-sonnet-4-6, claude-opus-4-6.
Requires lc0 servers running on ports 7100-7107.

Usage:
    source .venv/bin/activate
    python scripts/test_providers.py
    python scripts/test_providers.py --models gpt-5 claude-sonnet-4-6
    python scripts/test_providers.py --skip-lc0-check   # if lc0 is down
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

from agents.chess_analyst import ChessAnalyst

# Sicilian Najdorf after 6.Bg5 — white has the initiative, clear test position
TEST_FEN = "rnbqkb1r/1p2pppp/p2p1n2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R w KQkq - 0 7"
TEST_QUERY = f"What is white's best plan here? {TEST_FEN}"

CONFIGS = [
    ("azure", "gpt-5"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-opus-4-6"),
]


def run_one(provider: str, model: str, lc0_urls: list[str], skip_lc0: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"  Provider: {provider}   Model: {model}")
    print(f"{'='*60}")

    try:
        analyst = ChessAnalyst(
            model=model,
            provider=provider,
            lc0_urls=lc0_urls,
        )
    except Exception as e:
        return {"provider": provider, "model": model, "ok": False, "error": str(e)}

    if not skip_lc0:
        try:
            h = analyst.lc0.health()
            print(f"[lc0] {h.get('engine_id', {}).get('name', '?')}  backend={h.get('backend')}")
        except Exception as e:
            return {"provider": provider, "model": model, "ok": False, "error": f"lc0 unreachable: {e}"}

    t0 = time.time()
    try:
        answer = analyst.run(TEST_QUERY, verbose=True)
        elapsed = time.time() - t0
        stats = analyst.last_stats
        return {
            "provider": provider,
            "model": model,
            "ok": True,
            "elapsed_s": round(elapsed, 1),
            "llm_rounds": stats.get("llm_rounds"),
            "tool_calls": stats.get("tool_calls"),
            "tool_breakdown": stats.get("tool_breakdown"),
            "forced": stats.get("forced"),
            "answer_chars": len(answer),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {"provider": provider, "model": model, "ok": False, "elapsed_s": round(elapsed, 1), "error": str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        metavar="PROVIDER:MODEL",
        help="Override test configs as provider:model pairs, e.g. azure:gpt-5 anthropic:claude-sonnet-4-6",
    )
    p.add_argument(
        "--lc0-urls",
        default=",".join(f"http://localhost:{7100 + i}" for i in range(8)),
    )
    p.add_argument(
        "--skip-lc0-check",
        action="store_true",
        help="Skip lc0 health check (use if servers aren't up yet)",
    )
    args = p.parse_args()

    lc0_urls = [u.strip() for u in args.lc0_urls.split(",") if u.strip()]

    configs = CONFIGS
    if args.configs:
        configs = []
        for s in args.configs:
            provider, model = s.split(":", 1)
            configs.append((provider, model))

    results = []
    for provider, model in configs:
        r = run_one(provider, model, lc0_urls, skip_lc0=args.skip_lc0_check)
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        label = f"{r['provider']}:{r['model']}"
        if r["ok"]:
            print(
                f"  [{status}] {label:<38}  "
                f"{r['elapsed_s']}s  "
                f"rounds={r['llm_rounds']}  "
                f"tools={r['tool_calls']}  "
                f"chars={r['answer_chars']}"
                + ("  [FORCED]" if r.get("forced") else "")
            )
        else:
            print(f"  [{status}] {label:<38}  {r.get('error', '')}")

    any_failed = any(not r["ok"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
