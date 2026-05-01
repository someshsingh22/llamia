"""Chess analyst agent — multi-provider LLM + lc0 engine tools.

Supports three providers:
  "vllm"      — local vLLM server (Qwen3.5 with XML tool-call fallback)
  "azure"     — AzureOpenAI (GPT-5 / o-series reasoning models)
  "azure" + claude model — Azure-hosted Claude via native Anthropic SDK
  "anthropic" — direct Anthropic API

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
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
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

# ── OpenAI-format tool definitions (vllm / azure-GPT) ─────────────────────────
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

# ── Anthropic-format tool definitions (claude on azure or direct anthropic) ───
TOOL_DEFS_ANTHROPIC = [
    {
        "name": "get_position",
        "description": "Return current FEN, side to move, legal moves, and recent move history.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "make_move",
        "description": "Apply a move in UCI (e2e4) or SAN (Nxd4) notation to the current board.",
        "input_schema": {
            "type": "object",
            "properties": {
                "move": {"type": "string", "description": "Move in UCI or SAN notation"}
            },
            "required": ["move"],
        },
    },
    {
        "name": "undo_move",
        "description": "Undo the last move.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "reset_position",
        "description": "Reset board to a given FEN string.",
        "input_schema": {
            "type": "object",
            "properties": {"fen": {"type": "string"}},
            "required": ["fen"],
        },
    },
    {
        "name": "analyze",
        "description": (
            "Run lc0 engine search on the current position, optionally AFTER "
            "applying a sequence of hypothetical moves (does NOT mutate state). "
            "Returns top moves with centipawn scores (positive = good for side to move) "
            "and SAN principal variations. "
            "Use `moves` to explore variations cheaply: e.g. analyze(moves=['Nxd4']) "
            "returns the engine view after Nxd4 in ONE call — no make_move/undo needed."
        ),
        "input_schema": {
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
    {
        "name": "get_policy",
        "description": "Get lc0 raw NN prior P and value V per move (minimal search, fast).",
        "input_schema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "integer", "description": "Node budget (default: auto)"}
            },
            "required": [],
        },
    },
]

# Regex to extract a FEN from natural language (query may embed one)
_FEN_RE = re.compile(
    r"[rnbqkpRNBQKP1-8]{1,8}(?:/[rnbqkpRNBQKP1-8]{1,8}){7}"
    r"\s+[wb]\s+[KQkq\-]+\s+(?:[a-h][36]|-)\s+\d+\s+\d+"
)

MAX_ROUNDS = 14

# Read-only tools that can be fanned out in parallel (no board state mutation)
READ_ONLY_TOOLS = {"analyze", "get_policy", "get_position"}


class ChessAnalyst:
    def __init__(
        self,
        llm_base_url: str = "http://localhost:7000/v1",
        model: str = "Qwen/Qwen3-4B",
        lc0_url: str = "http://localhost:7100",
        initial_fen: str | None = None,
        system_prompt: str | None = None,
        provider: str = "vllm",
        reasoning_effort: str | None = None,
    ):
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.lc0 = LcOClient(base_url=lc0_url)
        self.state = ChessState(initial_fen or chess.STARTING_FEN)
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.last_stats: dict = {}
        self.last_trace: list[dict] = []

        # Determine whether to use the Anthropic SDK path
        is_claude = model.startswith("claude")
        self._use_anthropic = (provider == "azure" and is_claude) or (provider == "anthropic")

        if self._use_anthropic:
            import anthropic  # local import to avoid hard dep when not used

            if provider == "anthropic":
                self.anth_client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
            else:
                # azure + claude — derive services.ai.azure.com endpoint
                endpoint = (
                    os.environ.get("AZURE_ANTHROPIC_BASE_URL")
                    or self._derive_anthropic_base_url(
                        os.environ.get(
                            "AZURE_OPENAI_API_ENDPOINT",
                            os.environ.get("AZURE_API_BASE", ""),
                        )
                    )
                )
                self.anth_client = anthropic.Anthropic(
                    base_url=endpoint,
                    api_key=os.environ.get("AZURE_API_KEY", ""),
                    default_headers={"api-key": os.environ.get("AZURE_API_KEY", "")},
                )
            self.client = None  # no OpenAI client in Anthropic path

        elif provider == "azure":
            self.client = openai.AzureOpenAI(
                azure_endpoint=os.environ.get(
                    "AZURE_OPENAI_API_ENDPOINT",
                    os.environ.get("AZURE_API_BASE", ""),
                ),
                api_key=os.environ.get("AZURE_API_KEY", ""),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            )
            self.anth_client = None

        else:
            # vllm (default)
            self.client = openai.OpenAI(base_url=llm_base_url, api_key="none")
            self.anth_client = None

    # ── Endpoint derivation ────────────────────────────────────────────────────
    @staticmethod
    def _derive_anthropic_base_url(cognitiveservices_endpoint: str) -> str:
        """Convert a cognitiveservices endpoint to an Anthropic services URL.

        Example:
            https://mdsr-foundry-resource.cognitiveservices.azure.com/
            → https://mdsr-foundry-resource.services.ai.azure.com/anthropic
        """
        m = re.match(
            r"https://([^.]+)\.cognitiveservices\.azure\.com",
            cognitiveservices_endpoint,
        )
        if m:
            name = m.group(1)
            return f"https://{name}.services.ai.azure.com/anthropic"
        # Fallback: return as-is (may already be the correct URL)
        return cognitiveservices_endpoint.rstrip("/")

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

    # ── Parallel execution helper ──────────────────────────────────────────────
    def _execute_calls(
        self,
        tool_calls: list[dict],
        verbose: bool,
    ) -> list[tuple[dict, Any]]:
        """Execute tool calls, fanning out read-only ones in parallel.

        Each element of `tool_calls` is a dict with keys ``name`` and
        ``arguments``.  Returns a list of ``(call_dict, result)`` pairs in the
        same order as ``tool_calls``.
        """
        ro_calls = [c for c in tool_calls if c["name"] in READ_ONLY_TOOLS]
        mut_calls = [c for c in tool_calls if c["name"] not in READ_ONLY_TOOLS]

        results: list[tuple[dict, Any]] = []

        if len(ro_calls) > 1:
            with ThreadPoolExecutor(max_workers=len(ro_calls)) as pool:
                futures = [
                    pool.submit(self._dispatch, c["name"], c["arguments"])
                    for c in ro_calls
                ]
                for c, fut in zip(ro_calls, futures):
                    results.append((c, fut.result()))
        else:
            for c in ro_calls:
                results.append((c, self._dispatch(c["name"], c["arguments"])))

        # Mutating calls run after reads, in the order the model emitted them.
        for c in mut_calls:
            results.append((c, self._dispatch(c["name"], c["arguments"])))

        # Re-sort to the model's original call order.
        order_idx = {id(c): i for i, c in enumerate(tool_calls)}
        results.sort(key=lambda pair: order_idx[id(pair[0])])

        if verbose:
            for call, result in results:
                name = call["name"]
                args = call["arguments"]
                arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                print(f"  ⚙  {name}({arg_str})")
                rs = json.dumps(result, indent=2)
                preview = rs[:500] + ("..." if len(rs) > 500 else "")
                print(f"     -> {preview}\n")

        return results

    # ── Public entry point ─────────────────────────────────────────────────────
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

        self._reset_locked = fen_loaded  # checked in _dispatch

        if self._use_anthropic:
            return self._run_anthropic_loop(query, fen_loaded=fen_loaded, verbose=verbose)
        else:
            return self._run_openai_loop(query, fen_loaded=fen_loaded, verbose=verbose)

    # ── OpenAI / vLLM / Azure-GPT loop ────────────────────────────────────────
    def _run_openai_loop(
        self,
        query: str,
        fen_loaded: bool,
        verbose: bool,
    ) -> str:
        is_vllm = self.provider == "vllm"

        # Drop reset_position when the FEN was already loaded from the query.
        active_tools = TOOL_DEFS if not fen_loaded else [
            t for t in TOOL_DEFS if t["function"]["name"] != "reset_position"
        ]

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        tool_call_count = 0
        llm_round_count = 0
        tool_breakdown: dict[str, int] = {}
        _trace: list[dict] = []

        for _round_idx in range(MAX_ROUNDS):
            llm_round_count += 1

            # Build kwargs — GPT-5 (azure) and vllm differ on several params
            call_kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                tools=active_tools,
                tool_choice="auto",
            )
            if self.provider == "azure":
                # GPT-5 / o-series: use max_completion_tokens, no temperature,
                # no extra_body (reasoning model rejects them)
                call_kwargs["max_completion_tokens"] = 16000
                if self.reasoning_effort:
                    call_kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                # vllm / local
                call_kwargs["max_tokens"] = 700
                call_kwargs["temperature"] = 0.0
                call_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

            resp = self.client.chat.completions.create(**call_kwargs)
            choice = resp.choices[0]
            raw_content: str = choice.message.content or ""

            # Print thinking if the reasoning parser exposes it
            thinking = getattr(choice.message, "reasoning_content", None)
            if thinking and verbose:
                print(
                    f"[Thinking] {thinking[:400]}"
                    f"{'...' if len(thinking) > 400 else ''}\n"
                )

            # Tool calls: standard OpenAI field first (works for GPT-5 and
            # Qwen3 + hermes parser), falling back to Qwen3.5 inline XML.
            tool_calls: list[dict] = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                    tool_calls.append({"name": tc.function.name, "arguments": args})
                visible_text = raw_content.strip()
            else:
                if is_vllm:
                    # XML fallback only for vllm/Qwen3.5
                    tool_calls = _parse_qwen_tool_calls(raw_content)
                    visible_text = _strip_tool_calls(raw_content)
                else:
                    visible_text = raw_content.strip()

            if visible_text and verbose:
                print(f"[LLM] {visible_text}\n")

            # Append assistant turn.  With OpenAI-format tool calls we must
            # echo them back as `tool_calls` (each paired with a tool_call_id)
            # so the chat template can match the subsequent role=tool replies.
            assistant_msg: dict = {"role": "assistant", "content": raw_content or ""}
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            _openai_tc_ids = (
                [tc.id for tc in choice.message.tool_calls]
                if choice.message.tool_calls else None
            )

            # No tool calls → final answer
            if not tool_calls:
                _trace.append({
                    "round": llm_round_count,
                    "thinking": thinking or "",
                    "text": visible_text or raw_content,
                    "tool_calls": [],
                })
                self.last_stats = {
                    "llm_rounds": llm_round_count,
                    "tool_calls": tool_call_count,
                    "tool_breakdown": tool_breakdown,
                    "forced": False,
                }
                self.last_trace = _trace
                if verbose:
                    print(f"\n{'─'*60}\n[Answer]\n{visible_text or raw_content}\n")
                return visible_text or raw_content

            # Execute with parallel fan-out for read-only tools
            pairs = self._execute_calls(tool_calls, verbose=verbose)

            tc_trace: list[dict] = []
            for idx, (call, result) in enumerate(pairs):
                name = call["name"]
                tool_call_count += 1
                tool_breakdown[name] = tool_breakdown.get(name, 0) + 1
                tc_trace.append({"name": name, "input": call["arguments"], "output": result})

                tool_msg: dict = {"role": "tool", "content": json.dumps(result)}
                if _openai_tc_ids and idx < len(_openai_tc_ids):
                    tool_msg["tool_call_id"] = _openai_tc_ids[idx]
                messages.append(tool_msg)

            _trace.append({
                "round": llm_round_count,
                "thinking": thinking or "",
                "text": visible_text,
                "tool_calls": tc_trace,
            })

        # Fallback: tool call rounds exhausted — force a plain-text answer.
        forced_kwargs: dict[str, Any] = dict(model=self.model, messages=messages + [{
            "role": "user",
            "content": (
                "You have collected all the engine data needed. "
                "Now write your complete analysis as plain text. No tool calls."
            ),
        }])
        if self.provider == "azure":
            forced_kwargs["max_completion_tokens"] = 16000
            if self.reasoning_effort:
                forced_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            forced_kwargs["max_tokens"] = 700
            forced_kwargs["temperature"] = 0.0
            forced_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        resp = self.client.chat.completions.create(**forced_kwargs)
        final = _strip_tool_calls(resp.choices[0].message.content or "")
        forced_thinking = getattr(resp.choices[0].message, "reasoning_content", None) or ""
        _trace.append({
            "round": llm_round_count + 1,
            "thinking": forced_thinking,
            "text": final,
            "tool_calls": [],
        })
        self.last_stats = {
            "llm_rounds": llm_round_count + 1,
            "tool_calls": tool_call_count,
            "tool_breakdown": tool_breakdown,
            "forced": True,
        }
        self.last_trace = _trace
        if verbose:
            print(f"\n{'─'*60}\n[Answer]\n{final}\n")
        return final

    # ── Anthropic / Azure-Claude loop ──────────────────────────────────────────
    def _run_anthropic_loop(
        self,
        query: str,
        fen_loaded: bool,
        verbose: bool,
    ) -> str:
        # Drop reset_position when the FEN was already loaded from the query.
        active_tools = TOOL_DEFS_ANTHROPIC if not fen_loaded else [
            t for t in TOOL_DEFS_ANTHROPIC if t["name"] != "reset_position"
        ]

        # Anthropic format: system goes as a top-level param, not in messages.
        messages: list[dict] = [
            {"role": "user", "content": query},
        ]

        tool_call_count = 0
        llm_round_count = 0
        tool_breakdown: dict[str, int] = {}
        _trace: list[dict] = []

        # Map effort label → thinking budget_tokens for Claude extended thinking.
        _EFFORT_BUDGET = {"low": 1024, "medium": 4096, "high": 8192}
        _thinking_budget = _EFFORT_BUDGET.get(self.reasoning_effort or "", 0) if self.reasoning_effort else 0
        # max_tokens must exceed budget; add 2048 for response text.
        _max_tokens = max(1024, _thinking_budget + 2048)

        def _anth_call(msgs: list[dict]) -> Any:
            kwargs: dict[str, Any] = dict(
                model=self.model,
                system=self.system_prompt,
                messages=msgs,
                tools=active_tools,
                max_tokens=_max_tokens,
            )
            if _thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": _thinking_budget}
                kwargs["temperature"] = 1.0  # required when extended thinking is on
            else:
                kwargs["temperature"] = 0.0
            return self.anth_client.messages.create(**kwargs)

        for _round_idx in range(MAX_ROUNDS):
            llm_round_count += 1

            resp = _anth_call(messages)

            # Content blocks: thinking, text, tool_use
            thinking_parts: list[str] = []
            text_parts: list[str] = []
            tool_use_blocks: list[Any] = []
            for block in resp.content:
                if block.type == "thinking":
                    thinking_parts.append(getattr(block, "thinking", ""))
                elif block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_use_blocks.append(block)

            thinking_text = "\n".join(thinking_parts).strip()
            visible_text = "\n".join(text_parts).strip()

            if thinking_text and verbose:
                print(
                    f"[Thinking] {thinking_text[:400]}"
                    f"{'...' if len(thinking_text) > 400 else ''}\n"
                )
            if visible_text and verbose:
                print(f"[LLM] {visible_text}\n")

            # Serialize assistant message as content block dicts so we can
            # replay it back in the messages list on the next turn.
            assistant_content: list[dict] = []
            for block in resp.content:
                if block.type == "thinking":
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": getattr(block, "thinking", ""),
                        "signature": getattr(block, "signature", ""),
                    })
                elif block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Done when end_turn or no tool calls requested
            if resp.stop_reason == "end_turn" or not tool_use_blocks:
                _trace.append({
                    "round": llm_round_count,
                    "thinking": thinking_text,
                    "text": visible_text,
                    "tool_calls": [],
                })
                self.last_stats = {
                    "llm_rounds": llm_round_count,
                    "tool_calls": tool_call_count,
                    "tool_breakdown": tool_breakdown,
                    "forced": False,
                }
                self.last_trace = _trace
                if verbose:
                    print(f"\n{'─'*60}\n[Answer]\n{visible_text}\n")
                return visible_text

            # Build the normalized tool_calls list for _execute_calls
            norm_calls: list[dict] = [
                {"name": b.name, "arguments": b.input or {}, "_anth_block": b}
                for b in tool_use_blocks
            ]

            pairs = self._execute_calls(norm_calls, verbose=verbose)

            # Accumulate stats and build tool-result user message
            tc_trace: list[dict] = []
            tool_result_content: list[dict] = []
            for call, result in pairs:
                name = call["name"]
                block = call["_anth_block"]
                tool_call_count += 1
                tool_breakdown[name] = tool_breakdown.get(name, 0) + 1
                tc_trace.append({"name": name, "input": call["arguments"], "output": result})

                tool_result_content.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

            _trace.append({
                "round": llm_round_count,
                "thinking": thinking_text,
                "text": visible_text,
                "tool_calls": tc_trace,
            })
            messages.append({"role": "user", "content": tool_result_content})

        # Fallback: rounds exhausted — ask for plain-text answer
        messages.append({
            "role": "user",
            "content": (
                "You have collected all the engine data needed. "
                "Now write your complete analysis as plain text. No tool calls."
            ),
        })
        resp = _anth_call(messages)
        final_thinking = "\n".join(
            getattr(b, "thinking", "") for b in resp.content if b.type == "thinking"
        ).strip()
        final_parts = [b.text for b in resp.content if b.type == "text"]
        final = "\n".join(final_parts).strip()
        _trace.append({
            "round": llm_round_count + 1,
            "thinking": final_thinking,
            "text": final,
            "tool_calls": [],
        })
        self.last_stats = {
            "llm_rounds": llm_round_count + 1,
            "tool_calls": tool_call_count,
            "tool_breakdown": tool_breakdown,
            "forced": True,
        }
        self.last_trace = _trace
        if verbose:
            print(f"\n{'─'*60}\n[Answer]\n{final}\n")
        return final
