"""Stateful chess board — accepts UCI or SAN, maintains FEN."""
from __future__ import annotations
import chess


class ChessState:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self._history: list[str] = []  # SAN

    # ------------------------------------------------------------------
    def make_move(self, move: str) -> dict:
        board = self.board
        m = None
        try:
            m = chess.Move.from_uci(move.strip())
            if m not in board.legal_moves:
                m = None
        except ValueError:
            pass
        if m is None:
            try:
                m = board.parse_san(move.strip())
            except ValueError as e:
                return {"error": f"Illegal move '{move}': {e}"}
        san = board.san(m)
        board.push(m)
        self._history.append(san)
        return {
            "move_played": san,
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "in_check": board.is_check(),
            "checkmate": board.is_checkmate(),
        }

    def undo_move(self) -> dict:
        if not self.board.move_stack:
            return {"error": "No moves to undo"}
        self.board.pop()
        if self._history:
            self._history.pop()
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
        }

    def reset(self, fen: str) -> dict:
        try:
            self.board = chess.Board(fen)
            self._history = []
            return {"fen": self.board.fen(), "status": "ok"}
        except Exception as e:
            return {"error": str(e)}

    def info(self) -> dict:
        b = self.board
        legal = sorted(b.san(m) for m in b.legal_moves)
        return {
            "fen": b.fen(),
            "ascii_board": str(b),  # 8x8 grid, white pieces uppercase, black lowercase
            "turn": "white" if b.turn == chess.WHITE else "black",
            "move_number": b.fullmove_number,
            "in_check": b.is_check(),
            "legal_moves": legal[:24],
            "total_legal": len(legal),
            "recent_moves": self._history[-6:],
        }

    @property
    def fen(self) -> str:
        return self.board.fen()
