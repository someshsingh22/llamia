"""Thin wrapper around an lc0 UCI subprocess.

One Lc0Engine owns a single lc0 process pinned to a single GPU. python-chess
handles the UCI handshake, and we add helpers for (a) bounded search and
(b) a 1-node "raw" call that emits VerboseMoveStats which we parse to recover
per-move NN priors (P) and root value (V).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import chess
import chess.engine

logger = logging.getLogger(__name__)

# Example line:
# info string e2e4   (322 ) N:    13 (+ 0) (P:  6.79%) (WL:  0.045) (D: 0.43)
#                    (M: 158.6) (Q:  0.0451) (U:  0.0143) (S:  0.0594) (V: 4.7%)
_VMS_LINE = re.compile(r"^info string (\S+)\s+\(.*$")
_KV = re.compile(r"\(([A-Za-z]+):\s*(-?\d+\.?\d*)%?\)")


@dataclass
class MoveStat:
    move: str
    n: int = 0
    p: float | None = None  # NN prior in [0,1]
    q: float | None = None  # mean action value in [-1,1]
    wl: float | None = None
    d: float | None = None  # draw prob
    v: float | None = None  # NN value at child (single eval) in [-1,1]
    u: float | None = None
    s: float | None = None
    m: float | None = None  # plies-to-mate estimate


def parse_verbose_move_stats(info_strings: list[str]) -> list[MoveStat]:
    out: list[MoveStat] = []
    for line in info_strings:
        m = _VMS_LINE.match(line)
        if not m:
            continue
        move = m.group(1)
        if move in {"node", "stoppers", "search"}:  # non-move info-strings
            continue
        try:
            chess.Move.from_uci(move)
        except (ValueError, AssertionError):
            continue
        kv = {k.lower(): float(v) for k, v in _KV.findall(line)}
        n_match = re.search(r"N:\s*(\d+)", line)
        out.append(
            MoveStat(
                move=move,
                n=int(n_match.group(1)) if n_match else 0,
                p=kv.get("p", None) / 100.0 if "p" in kv else None,
                q=kv.get("q"),
                wl=kv.get("wl"),
                d=kv.get("d"),
                v=kv.get("v") / 100.0 if "v" in kv else None,
                u=kv.get("u"),
                s=kv.get("s"),
                m=kv.get("m"),
            )
        )
    return out


class Lc0Engine:
    """Single lc0 process. Methods are serialized via an internal lock —
    UCI is single-conversation, so callers above must not assume parallelism."""

    def __init__(
        self,
        binary: str,
        weights: str,
        backend: str = "cuda-fp16",
        threads: int = 2,
        extra_options: dict[str, object] | None = None,
    ) -> None:
        self.binary = binary
        self.weights = weights
        self.backend = backend
        self._lock = threading.Lock()
        cmd = [binary, f"--weights={weights}", f"--backend={backend}"]
        logger.info("starting lc0: %s", " ".join(cmd))
        self._engine = chess.engine.SimpleEngine.popen_uci(cmd)
        opts: dict[str, object] = {
            "Threads": threads,
            "VerboseMoveStats": True,
        }
        if extra_options:
            opts.update(extra_options)
        # python-chess auto-manages MultiPV per call; rejecting it here.
        opts.pop("MultiPV", None)
        # Filter to options the engine actually advertises (lc0 versions vary).
        advertised = set(self._engine.options.keys())
        applied = {k: v for k, v in opts.items() if k in advertised}
        if applied:
            self._engine.configure(applied)
        skipped = [k for k in opts if k not in advertised]
        if skipped:
            logger.info("skipped unknown UCI options: %s", skipped)
        self.id = self._engine.id

    def close(self) -> None:
        with self._lock:
            try:
                self._engine.quit()
            except Exception:  # noqa: BLE001
                pass

    # ---- helpers ------------------------------------------------------------

    def _limit(self, nodes: int | None, movetime_ms: int | None) -> chess.engine.Limit:
        if nodes is None and movetime_ms is None:
            nodes = 1
        if movetime_ms is not None:
            return chess.engine.Limit(time=movetime_ms / 1000.0)
        return chess.engine.Limit(nodes=nodes)

    @staticmethod
    def _build_board(fen: str, moves: list[str] | None) -> chess.Board:
        """Build a board from a FEN, optionally replaying UCI moves.

        Replaying matters for the BT4 net: it consumes 7 history planes plus the
        current position. With only a FEN, lc0 synthesises history (HistoryFill);
        with a real move stack the policy/value heads see the true game.
        """
        board = chess.Board(fen)
        for uci in moves or []:
            board.push_uci(uci)
        return board

    def analyse(
        self,
        fen: str,
        nodes: int | None = None,
        movetime_ms: int | None = None,
        multipv: int = 1,
        moves: list[str] | None = None,
    ) -> dict:
        board = self._build_board(fen, moves)
        limit = self._limit(nodes, movetime_ms)
        with self._lock:
            infos = self._engine.analyse(
                board, limit, multipv=multipv, info=chess.engine.INFO_ALL
            )
        if not isinstance(infos, list):
            infos = [infos]
        out_pvs = []
        for info in infos:
            score = info.get("score")
            pov = score.pov(board.turn) if score is not None else None
            out_pvs.append(
                {
                    "multipv": info.get("multipv", 1),
                    "score_cp": pov.score(mate_score=100000) if pov else None,
                    "mate": pov.mate() if pov else None,
                    "depth": info.get("depth"),
                    "seldepth": info.get("seldepth"),
                    "nodes": info.get("nodes"),
                    "nps": info.get("nps"),
                    "time_ms": int(info.get("time", 0) * 1000) if info.get("time") else None,
                    "pv": [m.uci() for m in info.get("pv", [])],
                }
            )
        return {
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "bestmove": out_pvs[0]["pv"][0] if out_pvs and out_pvs[0]["pv"] else None,
            "multipv": out_pvs,
        }

    def policy(
        self,
        fen: str,
        nodes: int | None = None,
        moves: list[str] | None = None,
        policy_temperature: float | None = None,
    ) -> dict:
        """Run a tiny search and parse VerboseMoveStats `info string` lines.

        With nodes ≥ #legal_moves, every child is expanded once and we get a
        per-move (P, V, N, Q) row — P/V come straight from the NN forward pass.

        For the *strongest raw policy* (top-1 = argmax(P)):
          - pass the full move history via `moves` so BT4's history planes are real
          - leave `policy_temperature` unset (None ⇒ use whatever the engine has);
            argmax does not depend on temperature, but rankings/distributions do.
        """
        board = self._build_board(fen, moves)
        n_legal = board.legal_moves.count()
        target = nodes if nodes is not None else max(n_legal + 2, 8)
        prev_temp: float | None = None
        if policy_temperature is not None and "PolicyTemperature" in self._engine.options:
            opt = self._engine.options["PolicyTemperature"]
            prev_temp = float(opt.default) if opt.default is not None else None
            with self._lock:
                self._engine.configure({"PolicyTemperature": str(policy_temperature)})
        captured: list[str] = []
        # We need the raw UCI info-string lines. python-chess strips them into
        # info["string"] one at a time during analysis. Collect via an analyser.
        with self._lock:
            with self._engine.analysis(
                board,
                chess.engine.Limit(nodes=target),
                multipv=1,
                info=chess.engine.INFO_ALL,
            ) as analysis:
                for info in analysis:
                    s = info.get("string")
                    if s:
                        captured.append("info string " + s)
                    if analysis.would_block():
                        # Drained current results; engine still running — keep iterating.
                        continue
        # Restore the engine's PolicyTemperature so subsequent calls aren't poisoned.
        if policy_temperature is not None and prev_temp is not None:
            with self._lock:
                self._engine.configure({"PolicyTemperature": str(prev_temp)})
        stats = parse_verbose_move_stats(captured)
        # Compute root value as N-weighted Q across children (proxy for V_root).
        total_n = sum(s.n for s in stats) or 1
        v_root = sum((s.q or 0.0) * s.n for s in stats) / total_n
        return {
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "nodes_searched": total_n,
            "value_root": v_root,
            "moves": [
                {
                    "move": s.move,
                    "P": s.p,
                    "N": s.n,
                    "Q": s.q,
                    "WL": s.wl,
                    "D": s.d,
                    "V": s.v,
                    "U": s.u,
                    "S": s.s,
                    "M": s.m,
                }
                for s in stats
            ],
            "raw_info_strings": captured if not stats else None,
        }
