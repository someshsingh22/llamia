"""VERL BaseTool wrappers for chess analysis (lc0-backed).

State management: each trajectory (agent_data.request_id) gets one ChessState
stored in agent_data.extra_fields["chess_board"]. FEN is extracted lazily from
the initial user message on first execute(), or from create_kwargs if the
dataset includes extra_info.tools_kwargs.

The tool_agent_loop calls create()+execute()+release() per tool invocation.
Each call gets a fresh instance_id, but agent_data (and its extra_fields)
persists across all tool calls in a trajectory — that is where board state lives.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional
from uuid import uuid4

import chess

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .state import ChessState
from .tools import LcOClient

_FEN_RE = re.compile(r"FEN:\s*(\S+)")

_LC0 = LcOClient(base_url="http://localhost:7100")

# instance_id -> initial_fen; populated in create(), cleared in release().
_INSTANCE_FEN: dict[str, str] = {}


def _extract_fen(instance_id: str, agent_data) -> str:
    """Return the initial FEN for this trajectory.

    Tries create_kwargs (populated if dataset has extra_info.tools_kwargs),
    then falls back to parsing the first user message that contains 'FEN: '.
    """
    if fen := _INSTANCE_FEN.get(instance_id):
        return fen
    for msg in agent_data.messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            m = _FEN_RE.search(content)
            if m:
                return m.group(1)
    return chess.STARTING_FEN


def _get_board(instance_id: str, agent_data) -> ChessState:
    """Get or lazily create the shared ChessState for this trajectory."""
    if "chess_board" not in agent_data.extra_fields:
        agent_data.extra_fields["chess_board"] = ChessState(_extract_fen(instance_id, agent_data))
    return agent_data.extra_fields["chess_board"]


class _ChessBaseTool(BaseTool):
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if fen := create_kwargs.get("fen"):
            _INSTANCE_FEN[instance_id] = fen
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        _INSTANCE_FEN.pop(instance_id, None)


class GetPositionTool(_ChessBaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        state = _get_board(instance_id, agent_data)
        return ToolResponse(text=json.dumps(state.info())), 0.0, {}


class MakeMoveTool(_ChessBaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        state = _get_board(instance_id, agent_data)
        result = state.make_move(parameters.get("move", ""))
        return ToolResponse(text=json.dumps(result)), 0.0, {}


class UndoMoveTool(_ChessBaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        state = _get_board(instance_id, agent_data)
        result = state.undo_move()
        return ToolResponse(text=json.dumps(result)), 0.0, {}


class AnalyzeTool(_ChessBaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        state = _get_board(instance_id, agent_data)
        moves_raw = parameters.get("moves")
        if isinstance(moves_raw, str):
            moves: list[str] | None = [m.strip() for m in moves_raw.split(",") if m.strip()] or None
        else:
            moves = moves_raw or None
        try:
            result = _LC0.analyze(
                state.fen,
                nodes=int(parameters.get("nodes", 800)),
                multipv=int(parameters.get("multipv", 3)),
                moves=moves,
            )
        except Exception as e:
            result = {"error": str(e)}
        return ToolResponse(text=json.dumps(result)), 0.0, {}


class GetPolicyTool(_ChessBaseTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        agent_data = kwargs["agent_data"]
        state = _get_board(instance_id, agent_data)
        nodes_raw = parameters.get("nodes")
        try:
            result = _LC0.get_policy(
                state.fen,
                nodes=int(nodes_raw) if nodes_raw is not None else None,
            )
        except Exception as e:
            result = {"error": str(e)}
        return ToolResponse(text=json.dumps(result)), 0.0, {}
