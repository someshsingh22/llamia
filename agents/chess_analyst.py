"""Chess analyst agent — Qwen3.5 LLM + lc0 engine tools.

Qwen3.5's chat template uses its own XML-like tool call format:
    <tool_call>
    <function=name>
    <parameter=key>value</parameter>
    </function>
    </tool_call>

vllm's hermes parser finds the <tool_call> tags but cannot parse the inner
XML as JSON, so msg.tool_calls is always empty and the raw XML stays in
msg.content.  We parse it ourselves with QWEN_TOOL_RE / PARAM_RE below.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

import chess
import openai

from .state import ChessState
from .tools import LcOClient

# ── Qwen3.5 tool-call format parsers ──────────────────────────────────────────
# Matches one complete tool call block (greedy over parameters inside)
QWEN_TOOL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)

# Matches trailing incomplete tool call (streaming / cut-off response)
QWEN_TOOL_OPEN_RE = re.compile(r"<tool_call>.*", re.DOTALL)


def _parse_qwen_tool_calls(content: str) -> list[dict]:
    """Extract [{name, arguments}] from Qwen3.5 XML tool-call blocks."""
    calls = []
    for m in QWEN_TOOL_RE.finditer(content):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for pm in PARAM_RE.finditer(body):
            val = pm.group(2).strip()
            # Try to coerce JSON-like values (numbers, booleans, null)
            try:
                args[pm.group(1)] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                args[pm.group(1)] = val
        calls.append({"name": name, "arguments": args})
    return calls


def _strip_tool_calls(content: str) -> str:
    """Remove tool call XML from visible content."""
    content = QWEN_TOOL_RE.sub("", content)
    content = QWEN_TOOL_OPEN_RE.sub("", content)
    return content.strip()


# ── Prompt / tool schema ───────────────────────────────────────────────────────
# Kept tight — 4 k token context window shared with tool outputs & responses.
SYSTEM_PROMPT = """\
You are a chess expert with access to lc0, a top neural network engine.
A board is already loaded — any FEN in the user's query has been applied for you.
DO NOT call reset_position unless you need a *different* position.

Tools (board is stateful across calls):
- get_position: FEN, ASCII board, legal moves, history
- make_move(move): apply UCI (e2e4) or SAN (Nf3) move PERMANENTLY
- undo_move: take back last move
- reset_position(fen): switch to a different FEN
- analyze(nodes, multipv, moves=[]): engine search. Pass `moves=[X]` to
  analyze the position AFTER playing X without changing state — use this
  for "what if?" questions instead of make_move/undo.
- get_policy(nodes): raw NN priors P and values V per move (fast, 1 pass)

Blunder analysis pattern (2 calls — issue them TOGETHER in one response):
  - analyze() — engine's best move + eval for the current position.
  - analyze(moves=["<candidate>"]) — eval after the candidate move.
Compare the two evaluations; the difference is the centipawn loss.
The PV from the second call starts with the opponent's refutation (SAN, e.g. Qxg5+).

EFFICIENCY: when two tool calls are independent (don't depend on each
other's results), emit BOTH in your same response. The harness dispatches
them in parallel — you save a full round-trip."""

# Tool definitions fed to the API (used by vllm to build the chat template's
# tool list, which teaches the model what functions exist and their parameters).
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": "Return current FEN, side to move, legal moves, and recent move history.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_move",
            "description": "Apply a move in UCI (e2e4) or SAN (Nxd4) notation to the current board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {"type": "string", "description": "Move in UCI or SAN notation"}
                },
                "required": ["move"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_move",
            "description": "Undo the last move.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_position",
            "description": "Reset board to a given FEN string.",
            "parameters": {
                "type": "object",
                "properties": {"fen": {"type": "string"}},
                "required": ["fen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze",
            "description": (
                "Run lc0 engine search on the current position, optionally AFTER "
                "applying a sequence of hypothetical moves (does NOT mutate state). "
                "Returns top moves with centipawn scores (positive = good for side to move) "
                "and SAN principal variations. "
                "Use `moves` to explore variations cheaply: e.g. analyze(moves=['Nxd4']) "
                "returns the engine view after Nxd4 in ONE call — no make_move/undo needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "integer", "description": "Search budget (default 800)"},
                    "multipv": {"type": "integer", "description": "Top moves to return (default 3)"},
                    "moves": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional UCI/SAN moves to apply before analyzing. State is unchanged.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Get lc0 raw NN prior P and value V per move (minimal search, fast).",
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "integer", "description": "Node budget (default: auto)"}
                },
                "required": [],
            },
        },
    },
]

# Regex to extract a FEN from natural language (query may embed one)
_FEN_RE = re.compile(
    r"[rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7}"
    r"\s+[wb]\s+[KQkq\-]+\s+(?:[a-h][36]|-)\s+\d+\s+\d+"
)

MAX_ROUNDS = 14


class ChessAnalyst:
    def __init__(
        self,
        llm_base_url: str = "http://localhost:7000/v1",
        model: str = "Qwen/Qwen3.5-122B-A10B-FP8",
        lc0_url: str = "http://localhost:7100",
        initial_fen: str | None = None,
    ):
        self.model = model
        self.client = openai.OpenAI(base_url=llm_base_url, api_key="none")
        self.lc0 = LcOClient(base_url=lc0_url)
        self.state = ChessState(initial_fen or chess.STARTING_FEN)
        # Per-run telemetry — populated by run()
        self.last_stats: dict = {}

    # ── Tool dispatch ──────────────────────────────────────────────────────────
    def _dispatch(self, name: str, args: dict) -> Any:
        if name == "get_position":
            return self.state.info()
        if name == "make_move":
            return self.state.make_move(args.get("move", ""))
        if name == "undo_move":
            return self.state.undo_move()
        if name == "reset_position":
            if getattr(self, "_reset_locked", False):
                return {
                    "noop": True,
                    "message": (
                        "The board is already loaded with the FEN from the user's "
                        "query. No need to reset. Proceed directly to analysis."
                    ),
                    "current_fen": self.state.fen,
                }
            return self.state.reset(args.get("fen", ""))
        if name == "analyze":
            return self.lc0.analyze(
                self.state.fen,
                nodes=int(args.get("nodes", 800)),
                multipv=int(args.get("multipv", 3)),
                moves=args.get("moves"),
            )
        if name == "get_policy":
            nodes = args.get("nodes")
            return self.lc0.get_policy(
                self.state.fen,
                nodes=int(nodes) if nodes is not None else None,
            )
        return {"error": f"Unknown tool: {name}"}

    # ── Agent loop ─────────────────────────────────────────────────────────────
    def run(self, query: str, verbose: bool = True) -> str:
        # Auto-detect and load FEN embedded in the query
        fen_loaded = False
        fen_match = _FEN_RE.search(query)
        if fen_match:
            fen = fen_match.group(0)
            r = self.state.reset(fen)
            if "error" not in r:
                fen_loaded = True
                if verbose:
                    print(f"[Board] Position set: {fen}\n")

        # Drop reset_position from the tool list when we already loaded a FEN.
        # Without it in the tool catalogue the model is far less likely to
        # propose it, and if it does (from training prior), the dispatcher
        # rejects it with a clear message instead of silently succeeding.
        active_tools = TOOL_DEFS if not fen_loaded else [
            t for t in TOOL_DEFS if t["function"]["name"] != "reset_position"
        ]
        self._reset_locked = fen_loaded  # checked in _dispatch

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        tool_call_count = 0
        llm_round_count = 0
        tool_breakdown: dict[str, int] = {}

        for round_idx in range(MAX_ROUNDS):
            llm_round_count += 1
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
                max_tokens=700,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = resp.choices[0]
            raw_content: str = choice.message.content or ""

            # Print thinking if the reasoning parser exposes it
            thinking = getattr(choice.message, "reasoning_content", None)
            if thinking and verbose:
                print(f"[Thinking] {thinking[:400]}{'…' if len(thinking) > 400 else ''}\n")

            # ── Parse Qwen3.5 tool calls from content ──────────────────────
            tool_calls = _parse_qwen_tool_calls(raw_content)
            visible_text = _strip_tool_calls(raw_content)

            if visible_text and verbose:
                print(f"[LLM] {visible_text}\n")

            # Append assistant turn (raw content preserved so the model sees
            # its own output correctly in subsequent turns)
            messages.append({"role": "assistant", "content": raw_content})

            # No tool calls → final answer
            if not tool_calls:
                self.last_stats = {
                    "llm_rounds": llm_round_count,
                    "tool_calls": tool_call_count,
                    "tool_breakdown": tool_breakdown,
                    "forced": False,
                }
                if verbose:
                    print(f"\n{'─'*60}\n[Answer]\n{visible_text or raw_content}\n")
                return visible_text or raw_content

            # Execute tool calls.  Pure-read, no-state-mutation tools
            # (analyze, get_policy, get_position) run in parallel via threads;
            # state-mutating tools (make_move, undo_move, reset_position)
            # MUST stay sequential because order matters.
            READ_ONLY_TOOLS = {"analyze", "get_policy", "get_position"}
            ro_calls = [c for c in tool_calls if c["name"] in READ_ONLY_TOOLS]
            mut_calls = [c for c in tool_calls if c["name"] not in READ_ONLY_TOOLS]

            results: list[tuple[dict, Any]] = []
            if len(ro_calls) > 1:
                # Fan-out: dispatch all read-only calls concurrently.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(ro_calls)) as pool:
                    futures = [pool.submit(self._dispatch, c["name"], c["arguments"])
                               for c in ro_calls]
                    for c, fut in zip(ro_calls, futures):
                        results.append((c, fut.result()))
            else:
                for c in ro_calls:
                    results.append((c, self._dispatch(c["name"], c["arguments"])))

            # Mutating calls run after the reads, in the order the model emitted them.
            for c in mut_calls:
                results.append((c, self._dispatch(c["name"], c["arguments"])))

            # Re-emit results in the model's original call order so the
            # tool-response messages line up with what it expects.
            order_idx = {id(c): i for i, c in enumerate(tool_calls)}
            results.sort(key=lambda pair: order_idx[id(pair[0])])

            for call, result in results:
                name = call["name"]
                args = call["arguments"]
                tool_call_count += 1
                tool_breakdown[name] = tool_breakdown.get(name, 0) + 1

                if verbose:
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    print(f"  ⚙  {name}({arg_str})")
                    rs = json.dumps(result, indent=2)
                    preview = rs[:500] + ("…" if len(rs) > 500 else "")
                    print(f"     → {preview}\n")

                # Return tool results in standard OpenAI format; vllm / the
                # Qwen chat template will format them correctly for the next turn.
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

        # Fallback: all tool call rounds spent — explicitly ask for the answer.
        messages.append({
            "role": "user",
            "content": (
                "You have collected all the engine data needed. "
                "Now write your complete analysis as plain text. No tool calls."
            ),
        })
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=700,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        final = _strip_tool_calls(resp.choices[0].message.content or "")
        self.last_stats = {
            "llm_rounds": llm_round_count + 1,
            "tool_calls": tool_call_count,
            "tool_breakdown": tool_breakdown,
            "forced": True,
        }
        if verbose:
            print(f"\n{'─'*60}\n[Answer]\n{final}\n")
        return final
