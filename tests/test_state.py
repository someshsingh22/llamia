"""Unit tests for ChessState — no network required."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import pytest
from agents.state import ChessState

STARTING_FEN = chess.STARTING_FEN
BLUNDER_FEN = "rnbqkb1r/pp2pppp/5n2/6B1/2pp4/4PN2/PP3PPP/RN1QKB1R w KQkq - 0 6"


# 1. Initial position is chess.STARTING_FEN
def test_initial_position():
    state = ChessState()
    assert state.board.fen() == STARTING_FEN


# 2. make_move with UCI ("e2e4") returns correct SAN ("e4")
def test_make_move_uci():
    state = ChessState()
    result = state.make_move("e2e4")
    assert result.get("move_played") == "e4"
    assert "error" not in result


# 3. make_move with SAN ("Nf3") works
def test_make_move_san():
    state = ChessState()
    result = state.make_move("Nf3")
    assert result.get("move_played") == "Nf3"
    assert "error" not in result


# 4. Illegal move returns {"error": ...} without raising
def test_illegal_move_returns_error():
    state = ChessState()
    result = state.make_move("e2e5")  # not a legal pawn move
    assert "error" in result


# 5. undo_move after one move restores STARTING_FEN
def test_undo_restores_starting_fen():
    state = ChessState()
    state.make_move("e2e4")
    result = state.undo_move()
    assert "error" not in result
    assert state.board.fen() == STARTING_FEN


# 6. undo_move on empty stack returns {"error": ...}
def test_undo_empty_stack():
    state = ChessState()
    result = state.undo_move()
    assert "error" in result


# 7. reset(fen) sets the board correctly
def test_reset_sets_board():
    state = ChessState()
    state.make_move("e2e4")
    result = state.reset(BLUNDER_FEN)
    assert "error" not in result
    assert state.board.fen() == chess.Board(BLUNDER_FEN).fen()


# 8. reset with invalid FEN returns {"error": ...}
def test_reset_invalid_fen():
    state = ChessState()
    result = state.reset("not-a-fen")
    assert "error" in result


# 9. info() returns correct turn, move_number, legal_moves
def test_info_starting_position():
    state = ChessState()
    info = state.info()
    assert info["turn"] == "white"
    assert info["move_number"] == 1
    assert isinstance(info["legal_moves"], list)
    assert len(info["legal_moves"]) > 0
    assert info["total_legal"] == 20  # standard chess starting position


# 10. Move history accumulates correctly (last 6 shown)
def test_move_history_accumulates():
    state = ChessState()
    # Legal 7-move sequence; d1h5 would be illegal here (Nf3 blocks the diagonal),
    # so we go via a different route. Each call must succeed.
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "b1c3"]
    for move in moves:
        result = state.make_move(move)
        assert "error" not in result, f"Move {move} unexpectedly illegal: {result}"
    info = state.info()
    # recent_moves shows last 6
    assert len(info["recent_moves"]) == 6
    # total history should have 7 entries
    assert len(state._history) == 7
    # Silent-failure guard: an illegal move must NOT extend the history.
    state.make_move("e2e4")  # already moved away, illegal here
    assert len(state._history) == 7


# 11. Checkmate detection: Scholar's mate variation
def test_checkmate_detection():
    state = ChessState()
    # 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
    moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
    for move in moves[:-1]:
        state.make_move(move)
    result = state.make_move(moves[-1])
    assert result.get("checkmate") is True


# 12. FEN round-trip: reset(board.fen()) → same position
def test_fen_round_trip():
    state = ChessState()
    state.make_move("e2e4")
    state.make_move("e7e5")
    current_fen = state.board.fen()
    state.reset(current_fen)
    assert state.board.fen() == current_fen


# 13. En-passant move works
def test_en_passant():
    # Set up en passant: white pawn on e5, black plays d7d5, then e5d6 is en passant
    state = ChessState()
    state.make_move("e2e4")
    state.make_move("a7a6")
    state.make_move("e4e5")
    state.make_move("d7d5")  # now e5d6 en passant is available
    result = state.make_move("e5d6")  # en passant capture
    assert "error" not in result, f"En passant failed: {result}"


# 14. Castling works
def test_castling():
    # Set up a position where white can castle kingside
    # Start FEN with cleared path for kingside castling
    fen = "rnbqkbnr/pppppppp/8/8/4P3/5NB1/PPPP1PPP/RNBQK2R w KQkq - 0 3"
    state = ChessState(fen)
    result = state.make_move("e1g1")  # kingside castling
    assert "error" not in result, f"Castling failed: {result}"
    assert result.get("move_played") == "O-O"


# 15. The blunder position: Nxd4 succeeds, then Qa5+ gives in_check=True
def test_blunder_position():
    state = ChessState(BLUNDER_FEN)
    # White plays Nxd4
    result = state.make_move("Nxd4")
    assert "error" not in result, f"Nxd4 failed: {result}"
    # Black plays Qa5+
    result = state.make_move("Qa5+")
    assert "error" not in result, f"Qa5+ failed: {result}"
    assert result.get("in_check") is True
