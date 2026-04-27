"""lc0 HTTP client with concise output formatting for the LLM context."""
from __future__ import annotations
import chess
import httpx


def _uci_pv_to_san(fen: str, uci_moves: list[str]) -> str:
    """Replay a UCI principal variation on a board and emit SAN.

    SAN shows captures (Qxg5), checks (+), mates (#), and disambiguates
    pieces, so the LLM cannot mistake e.g. `a5g5` for a quiet move when it
    is actually `Qxg5+`.  Falls back to UCI for any move that fails to parse
    (engine PVs are usually clean, but we don't want to crash on edge cases).
    """
    board = chess.Board(fen)
    out: list[str] = []
    for u in uci_moves:
        try:
            m = chess.Move.from_uci(u)
            if m not in board.legal_moves:
                out.append(u)
                break
            out.append(board.san(m))
            board.push(m)
        except (chess.InvalidMoveError, ValueError, AssertionError):
            out.append(u)
            break
    return " ".join(out)


class LcOClient:
    def __init__(self, base_url: str = "http://localhost:7100"):
        self._url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=60.0)

    def health(self) -> dict:
        return self._http.get(f"{self._url}/health").raise_for_status().json()

    def analyze(
        self,
        fen: str,
        nodes: int = 800,
        multipv: int = 3,
        moves: list[str] | None = None,
    ) -> dict:
        """Analyze a position, optionally after playing a sequence of moves.

        `moves` is a list of UCI or SAN move strings to apply on top of the
        FEN before analysis.  This is a no-mutate primitive: the agent's
        ChessState is untouched.  Lets the model ask "what's the eval after
        Nxd4?" in one call instead of make_move → analyze → undo.
        """
        # Resolve the analysis FEN: apply any candidate moves client-side.
        analysis_fen = fen
        applied_san: list[str] = []
        if moves:
            board = chess.Board(fen)
            for mv in moves:
                m = None
                try:
                    m = chess.Move.from_uci(mv.strip())
                    if m not in board.legal_moves:
                        m = None
                except ValueError:
                    pass
                if m is None:
                    try:
                        m = board.parse_san(mv.strip())
                    except ValueError:
                        return {"error": f"Illegal move in `moves`: {mv!r}"}
                applied_san.append(board.san(m))
                board.push(m)
            analysis_fen = board.fen()

        r = self._http.post(
            f"{self._url}/analyze",
            json={"fen": analysis_fen, "nodes": nodes, "multipv": multipv},
        )
        r.raise_for_status()
        data = r.json()
        lines = []
        for pv in data.get("multipv", []):
            if pv.get("mate") is not None:
                score = f"M{pv['mate']}"
            else:
                score = f"{(pv.get('score_cp') or 0) / 100:+.2f}"
            uci_moves = pv.get("pv", [])[:6]
            san_pv = _uci_pv_to_san(analysis_fen, uci_moves)
            lines.append(
                f"{pv['multipv']}. {san_pv}  score={score}  depth={pv.get('depth')}"
            )
        best_san = None
        if data.get("bestmove"):
            try:
                bm = chess.Move.from_uci(data["bestmove"])
                bd = chess.Board(analysis_fen)
                if bm in bd.legal_moves:
                    best_san = bd.san(bm)
            except (chess.InvalidMoveError, ValueError):
                pass

        out = {
            "turn": data.get("turn"),
            "bestmove": best_san or data.get("bestmove"),
            "lines": lines,
        }
        if applied_san:
            out["applied_moves"] = applied_san
        return out

    def get_policy(self, fen: str, nodes: int | None = None) -> dict:
        payload: dict = {"fen": fen}
        if nodes is not None:
            payload["nodes"] = nodes
        r = self._http.post(f"{self._url}/policy", json=payload)
        r.raise_for_status()
        data = r.json()
        moves = sorted(
            data.get("moves", []),
            key=lambda m: m.get("P") or 0,
            reverse=True,
        )[:12]
        rows = []
        for m in moves:
            p = f"{(m['P'] or 0)*100:.1f}%" if m.get("P") is not None else "?"
            v = f"{m['V']:+.3f}" if m.get("V") is not None else "?"
            q = f"{m['Q']:+.3f}" if m.get("Q") is not None else "?"
            rows.append(f"{m['move']}  P={p}  V={v}  Q={q}  N={m.get('N', 0)}")
        return {
            "turn": data.get("turn"),
            "value_root": round(data.get("value_root") or 0, 4),
            "nodes_searched": data.get("nodes_searched"),
            "top_moves": rows,
        }

    def close(self):
        self._http.close()
