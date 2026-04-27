#!/usr/bin/env bash
# Convenience wrapper for bench_elo.py.
# Defaults: 50 games, lc0 at 1000 nodes, Stockfish at depth 8 (~3000 Elo).
#
# Run on an idle GPU (the match takes a few minutes; lc0 holds ~3 GB).
# Stockfish must be in PATH (apt-get install stockfish, or pass --ref-engine).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
exec "$REPO_ROOT/lc0_server/.venv/bin/python" "$ROOT/scripts/bench_elo.py" \
    --games "${GAMES:-50}" \
    --lc0-nodes "${LC0_NODES:-1000}" \
    --ref-depth "${REF_DEPTH:-8}" \
    "$@"
