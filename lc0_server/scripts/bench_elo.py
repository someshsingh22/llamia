"""Calibration match: lc0 (BT4 + our settings) versus a reference engine.

Plays N games, alternating colors, with light random opening to avoid book
overfitting. Reports W/D/L and an Elo difference with 95 % CI.

USAGE
-----
    # Reference defaults to Stockfish in $PATH at depth 8 single-thread.
    python lc0_server/scripts/bench_elo.py --games 50

    # Custom: lc0 nodes=2000 vs Stockfish depth=12 over 100 games on GPU 0
    CUDA_VISIBLE_DEVICES=0 python lc0_server/scripts/bench_elo.py \\
        --games 100 --lc0-nodes 2000 --ref-depth 12 \\
        --ref-engine /usr/games/stockfish

The script does NOT speak HTTP — it spawns its own lc0 process so the match
isolates engine strength from server-side overhead. Settings mirror the
production server (MinibatchSize=128, cuda-fp16, BT4 weights, all search
flags at upstream defaults).

INTERPRETING THE RESULT
-----------------------
Elo offset = -400 * log10((L + D/2) / (W + D/2)) for the reference engine.
A reference Stockfish-17 NNUE on `--ref-depth 8` is roughly 3000 CCRL Elo;
`--ref-depth 12` is ~3300. So a measured +200 Elo against depth-8 SF would
place lc0-at-our-settings near 3200.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LC0_BIN = "/dev/shm/somesh/lc0_src/build/release/lc0"
DEFAULT_LC0_NET = REPO_ROOT / "lc0_server/weights/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"

# Twelve common opening positions (white to move after 2-3 ply). Keeps games
# diverse without a book file. Each is played twice (once per color).
OPENING_PLIES = [
    [],
    ["e2e4", "e7e5"],
    ["e2e4", "c7c5"],
    ["e2e4", "e7e6"],
    ["e2e4", "c7c6"],
    ["d2d4", "d7d5"],
    ["d2d4", "g8f6"],
    ["d2d4", "d7d5", "c2c4"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["c2c4", "e7e5"],
    ["g1f3", "d7d5"],
    ["e2e4", "d7d6"],
]


def open_lc0(args) -> chess.engine.SimpleEngine:
    cmd = [args.lc0_bin, f"--weights={args.lc0_net}", f"--backend={args.lc0_backend}"]
    eng = chess.engine.SimpleEngine.popen_uci(cmd)
    eng.configure({
        "Threads": args.lc0_threads,
        "MinibatchSize": args.lc0_minibatch,
        "MaxPrefetch": args.lc0_max_prefetch,
        "NNCacheSize": args.lc0_nncache,
    })
    return eng


def open_reference(args) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci([args.ref_engine])
    cfg = {"Threads": args.ref_threads, "Hash": args.ref_hash_mb}
    advertised = set(eng.options.keys())
    eng.configure({k: v for k, v in cfg.items() if k in advertised})
    return eng


def play_game(white, white_limit, black, black_limit, plies):
    """Play one game; return result in {1, 0, 0.5} from White's perspective."""
    board = chess.Board()
    for uci in plies:
        board.push_uci(uci)
    while not board.is_game_over(claim_draw=True):
        eng, lim = (white, white_limit) if board.turn == chess.WHITE else (black, black_limit)
        result = eng.play(board, lim)
        if result.move is None:
            break
        board.push(result.move)
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def elo_from_score(wins: float, draws: float, losses: float) -> tuple[float, float]:
    """Returns (elo_diff, 95% CI half-width)."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, float("inf")
    score = (wins + 0.5 * draws) / n
    if score in (0.0, 1.0):
        return float("inf") if score == 1.0 else float("-inf"), float("inf")
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # SE on score, mapped to Elo via local derivative of the logistic.
    p = score
    se_score = math.sqrt(p * (1 - p) / n)
    deriv = 400.0 / math.log(10) / max(p * (1 - p), 1e-9)
    return elo, 1.96 * se_score * deriv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--lc0-bin", default=DEFAULT_LC0_BIN)
    p.add_argument("--lc0-net", default=str(DEFAULT_LC0_NET))
    p.add_argument("--lc0-backend", default="cuda-fp16")
    p.add_argument("--lc0-threads", type=int, default=2)
    p.add_argument("--lc0-minibatch", type=int, default=128)
    p.add_argument("--lc0-max-prefetch", type=int, default=32)
    p.add_argument("--lc0-nncache", type=int, default=200000)
    p.add_argument("--lc0-nodes", type=int, default=1000)

    p.add_argument("--ref-engine", default=shutil.which("stockfish") or "stockfish")
    p.add_argument("--ref-threads", type=int, default=1)
    p.add_argument("--ref-hash-mb", type=int, default=128)
    p.add_argument("--ref-depth", type=int, default=8,
                   help="Reference search depth (Stockfish: ~3000 Elo at depth 8)")

    args = p.parse_args()

    if not Path(args.lc0_bin).exists():
        print(f"ERROR: lc0 binary not found at {args.lc0_bin}", file=sys.stderr)
        return 2
    if not Path(args.lc0_net).exists():
        print(f"ERROR: lc0 net not found at {args.lc0_net}", file=sys.stderr)
        return 2
    if not (Path(args.ref_engine).exists() or shutil.which(args.ref_engine)):
        print(f"ERROR: reference engine '{args.ref_engine}' not found in PATH or filesystem.\n"
              f"  Install: apt-get install stockfish  (or)  conda install -c conda-forge stockfish",
              file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    lc0 = open_lc0(args)
    ref = open_reference(args)
    lc0_lim = chess.engine.Limit(nodes=args.lc0_nodes)
    ref_lim = chess.engine.Limit(depth=args.ref_depth)

    print(f"# lc0 nodes={args.lc0_nodes} backend={args.lc0_backend} minibatch={args.lc0_minibatch}")
    print(f"# ref engine={args.ref_engine} depth={args.ref_depth} threads={args.ref_threads}")
    print(f"# games={args.games}")
    print()

    w = d = l = 0
    t0 = time.time()
    try:
        for i in range(args.games):
            opening = list(rng.choice(OPENING_PLIES))
            lc0_white = (i % 2 == 0)
            white, w_lim = (lc0, lc0_lim) if lc0_white else (ref, ref_lim)
            black, b_lim = (ref, ref_lim) if lc0_white else (lc0, lc0_lim)
            score_white = play_game(white, w_lim, black, b_lim, opening)
            score_lc0 = score_white if lc0_white else (1.0 - score_white)
            if score_lc0 == 1.0:
                w += 1; tag = "W"
            elif score_lc0 == 0.0:
                l += 1; tag = "L"
            else:
                d += 1; tag = "D"
            elapsed = time.time() - t0
            print(f"[{i+1:>3d}/{args.games}] {'lc0=W' if lc0_white else 'lc0=B'} -> {tag} "
                  f"| running W/D/L = {w}/{d}/{l}  ({elapsed:.0f}s)")
    finally:
        lc0.quit(); ref.quit()

    print()
    n = w + d + l
    score = (w + 0.5 * d) / max(n, 1)
    elo, ci = elo_from_score(w, d, l)
    print(f"Final: W/D/L = {w}/{d}/{l}  score = {score*100:.1f}%")
    print(f"Elo(lc0) - Elo(ref) = {elo:+.0f}  ±{ci:.0f}  (95% CI)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
