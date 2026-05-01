# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**LLAMIA** — research on LLM ↔ specialist-agent collaboration, with chess as the testbed (decades of human-engine collaboration provide well-defined tasks, evaluation protocols, and pretrained engines with extractable neural representations).

Two artifacts:

- **LLAMIA-Bench** — a suite of collaborative tasks (commentary, behavioral analysis, difficulty assessment, strategic planning, etc.) stressing different facets of sustained LLM-agent interaction. Used to show that *verbalized* agent access — even with RL-optimized invocation — underperforms on tasks requiring sustained collaboration.
- **Latent state internalization** — the proposed integration paradigm. The agent's continuous representations are projected into `K=16` tokens placed inside the LLM's chain of thought, with **dynamic re-encoding** as the LLM's actions evolve the environment state. A single 13B model with internalization matches/exceeds task-specific specialists and outperforms much larger text-mediated LLMs.

Key ablation result: removing dynamic re-encoding causes large degradation on multi-step planning but only minor degradation on single-step prediction — the verbalization bottleneck compounds over sequential interactions. Keep this framing when reasoning about design choices: anything that breaks the *sustained, dynamically re-encoded* loop is the thing the project is fighting against.

## Repository

Owner: [someshsingh22](https://github.com/someshsingh22). Default branch: `main`. `.claude/` and `.remember/` are gitignored (local Claude Code state).

This file will grow as code, datasets, and tooling land — update it when concrete commands (training, eval, bench harness) and architecture (projector module, re-encoding hooks, engine adapters) are added.

## Paper (llamiaTex submodule)

The paper lives in `llamiaTex/` (a git submodule). Target venue: **ICML 2026**, using `icml2026.sty`.

### LaTeX structure

```
llamiaTex/
├── ICML.tex               # root — preamble, title, abstract, \input chain
├── math_commands.tex      # shared math macros (\vx, \vh, etc.)
├── sample.bib             # bibliography
├── pages/
│   ├── intro.tex          # §1 Introduction
│   ├── bg.tex             # §2 Background (MDP formalism, agents)
│   ├── methodology.tex    # §3 LLAMIA framework + SALT training
│   ├── experiment.tex     # §4 Experiments & evaluation
│   ├── results.tex        # additional results/tables (imported as needed)
│   ├── dataset.tex        # dataset description
│   ├── extension.tex      # extensions beyond chess
│   ├── appendix.tex       # appendix root — inputs appendix/ subtree
│   └── appendix/
│       ├── ablations.tex  # ablation tables
│       ├── dataset.tex    # dataset details
│       ├── examples.tex   # qualitative examples
│       ├── extension.tex  # extension details
│       ├── limitations.tex
│       ├── prompts.tex    # prompt listings
│       ├── study.tex      # human study details
│       └── details.tex    # ← implementation details (hyperparams, libraries, RL config)
└── figures/               # PDF/PNG figures
```

### Implementation details appendix

All reproducibility information — training hyperparameters, prompt templates, library versions, RL configuration, and dataset construction details — goes in `llamiaTex/pages/appendix/details.tex`. It is `\input`-ed from `appendix.tex`. When adding a new experiment or prompt, record the relevant details there immediately.

## Environments

Two completely isolated UV environments — no shared deps, no cross-contamination.

| | Root (VERL/training) | `lc0_server/` (engine server) |
|---|---|---|
| **pyproject.toml** | `pyproject.toml` | `lc0_server/pyproject.toml` |
| **venv** | `.venv/` | `lc0_server/.venv/` |
| **activate** | `source .venv/bin/activate` | `source lc0_server/.venv/bin/activate` |
| **install/update** | `uv sync` | `cd lc0_server && uv sync` |
| **key deps** | torch 2.7.0+cu126, verl 0.7.1, vllm 0.14.0, ray 2.55.1, chess, httpx | chess, fastapi, uvicorn, pydantic, httpx |

The engine server has **no torch** — it wraps the lc0 binary via subprocess (UCI protocol). The training env calls engine servers over HTTP using `httpx`; it never imports engine code directly.

Adding a new engine (e.g. Stockfish, a Go engine): create `<engine>_server/` with its own `pyproject.toml` + `.venv` + `scripts/start_<engine>_server.sh`, following the `lc0_server/` pattern.

- **Package manager**: `uv` (v0.10.12)
- **Python**: 3.10.15 via conda at `/opt/conda`
- **CUDA**: 12.6 | **GPU**: A100-SXM4-80GB ×8

## External vLLM Server

Already running on port 7000 (served by colligo):
```
Qwen/Qwen3.5-122B-A10B-FP8
--port 7000 --max-model-len 4000 --enable-expert-parallel
--reasoning-parser qwen3 --tool-call-parser hermes
--enable-auto-tool-choice --enable-prefix-caching
-dp 8 --max-num-batched-tokens 32768
```

## Lc0 Inference Server

Chess specialist used as the latent-state donor. lc0 v0.32.1 is built from source at
`/dev/shm/somesh/lc0_src/build/release/lc0` (CUDA 12.4, cuda+cuda-fp16 backends).
Weights for the BT4 transformer net (1024×15×32, ~382 MB) live under
`lc0_server/weights/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz`.

`lc0_server/server/` is a FastAPI wrapper around an lc0 UCI subprocess:

- `POST /analyze` — search-based: `{fen, moves?, nodes?, movetime_ms?, multipv?}` →
  `{bestmove, multipv:[{score_cp, mate, depth, nodes, time_ms, pv}]}`
- `POST /policy` — `{fen, moves?, nodes?, policy_temperature?}` runs a tiny search
  with `VerboseMoveStats=true` and parses per-move `(P, N, Q, WL, D, V, U, S, M)`
  from the `info string` lines. With `nodes ≥ #legal_moves` every child is
  expanded once, giving direct NN priors `P` and per-move single-eval values `V`.
  For the strongest raw policy, pass the full `moves` list (UCI strings) so BT4's
  7 history planes are real, and use `nodes=1` — `argmax(moves, key=P)` is then
  the unmodified NN top move.
- `GET /health`

Defaults tuned for A100 throughput (resident memory is flat at ~3 GB across all
batch sizes — see backendbench): `MinibatchSize=128` (~7 300 nps, 3× over =8),
`MaxPrefetch=32`, `NNCacheSize=200 000`. Override via `LC0_MINIBATCH`,
`LC0_MAX_PREFETCH`, `LC0_NNCACHE`.

Start the server (binds 0.0.0.0:7100, uses GPU 5 by default — vLLM has dp=8 so all
GPUs are partially loaded; GPU 5 has the most headroom):

```bash
./lc0_server/scripts/start_lc0_server.sh
# overrides:
CUDA_VISIBLE_DEVICES=2 LC0_PORT=7101 LC0_BACKEND=cuda \
  ./lc0_server/scripts/start_lc0_server.sh
```

For the latent-state work, `/policy` is the primary endpoint — it surfaces NN
priors and root value from a single forward pass. To eventually pull *hidden*
representations (the "K=16 projected tokens" path), the binary UCI route is not
enough; we'll need either `lczero-training`'s python bindings or our own
forward-pass shim that loads the same `.pb.gz` weights. The HTTP server is the
fast path for getting LLAMIA-Bench data flowing.

Calibration: `lc0_server/scripts/bench_elo.py` plays N games (default 50) against
Stockfish at fixed depth (default 8 ≈ 3000 CCRL Elo) and prints W/D/L + Elo with
95 % CI. Stockfish must be installed (`apt-get install stockfish`). Don't run
during training — it holds ~3 GB on whichever GPU you put it on.

## Chess Analyst Agent (inference)

`agents/` is the LLM + tool-call harness for inference-time chess analysis. It lives in the root `.venv`.

```
agents/
├── state.py          # ChessState — FEN board, accepts UCI or SAN moves
├── tools.py          # LcOClient — thin httpx wrapper around lc0_server HTTP API
└── chess_analyst.py  # ChessAnalyst — agentic loop (LLM + tool dispatch)
```

`ChessAnalyst` supports multiple LLM providers via `provider=` constructor arg and `--provider` CLI flag:

| `provider` | `model` | Auth | Notes |
|---|---|---|---|
| `vllm` (default) | `Qwen/Qwen3-4B` | none | Local vLLM server; Qwen3.5 XML tool-call parsing |
| `azure` | `gpt-5` | `AZURE_API_KEY`, `AZURE_OPENAI_API_ENDPOINT`, `AZURE_OPENAI_API_VERSION` | GPT-5/o-series: `max_completion_tokens=16000`, no temperature |
| `azure` | `claude-sonnet-4-6`, `claude-opus-4-6` | same Azure creds | Claude via native Anthropic SDK; base URL auto-derived: `{resource}.services.ai.azure.com/anthropic` |
| `anthropic` | `claude-*` | `ANTHROPIC_API_KEY` | Direct Anthropic API |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` | Standard OpenAI |

**Run:**
```bash
source .venv/bin/activate
# vLLM (default)
python scripts/run_analyst.py "Why is Nxd4 a blunder? rnbqkb1r/pp2pppp/5n2/6B1/2pp4/4PN2/PP3PPP/RN1QKB1R w KQkq - 0 6"
# Azure GPT-5
python scripts/run_analyst.py "Best plan?" --provider azure --model gpt-5
# Azure Claude (native Anthropic SDK, auto-routes to services.ai.azure.com/anthropic)
python scripts/run_analyst.py "Best plan?" --provider azure --model claude-sonnet-4-6
```

**Provider implementation details:**
- **vLLM/Qwen**: `choice.message.tool_calls` checked first (Qwen3 + hermes parser), XML fallback for Qwen3.5. `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. `max_tokens=700`, `temperature=0`.
- **Azure GPT-5/o-series**: `AzureOpenAI` client. `max_completion_tokens=16000` (internal reasoning consumes the budget — 700 tokens always exhausted). No `temperature` param (rejected). Standard `message.tool_calls` JSON.
- **Azure Claude**: `anthropic.Anthropic` SDK. Separate agent loop: system prompt as top-level param, tool defs in `input_schema` format, tool results in user messages as `tool_result` blocks. `max_tokens=1024`, `temperature=0`. Base URL: `https://{resource}.services.ai.azure.com/anthropic` (auto-derived from `AZURE_OPENAI_API_ENDPOINT`; override with `AZURE_ANTHROPIC_BASE_URL`).

**Reasoning effort control** (`reasoning_effort=` constructor arg):
- **GPT-5**: passes `reasoning_effort="low"|"medium"|"high"` directly to the Azure API
- **Claude**: maps to extended thinking budget — low→1024 tokens, medium→4096, high→8192; sets `thinking={"type":"enabled","budget_tokens":N}` and `temperature=1.0` (required)
- Thinking/reasoning content is captured per-turn in `analyst.last_trace`

**Raw-output eval** (`scripts/eval_raw.py`):
```bash
# Edit EVAL_CONFIG dict at top of the script to select models/efforts, then:
python scripts/eval_raw.py --n 100 --workers 8   # 8 workers → 8 lc0 servers on GPUs 0-7
python scripts/eval_raw.py --labels gpt-5-low claude-sonnet-4-6-medium --n 50
# Output JSONL in results/raw/<label>_<timestamp>.jsonl
# Each row: idx, label, model, effort, fen, pred/true pop+elo, trace[]
# trace[i]: {round, thinking, text, tool_calls:[{name, input, output}]}
```

Each worker gets its own dedicated lc0 server (started automatically on GPU i, port 7100+i).
`--lc0-base-port` overrides the starting port (default 7100).

**Eval on benchmark (legacy, no raw traces):**
```bash
python scripts/eval_providers.py --providers azure:gpt-5 azure:claude-sonnet-4-6 --n 100 --workers 3
# Results saved to results/<provider>_<model>_<timestamp>.jsonl
```

## RL Training (VERL)

Config: `configs/qwen3_ppo_vllm.yaml`

```bash
VLLM_HOST=localhost VLLM_PORT=7000 ./scripts/train_ppo.sh
# or with overrides:
python -m verl.trainer.main_ppo \
  --config-path configs --config-name qwen3_ppo_vllm \
  data.train_files=data/train.parquet
```

### Toy task: puzzle popularity + ELO

End-to-end GRPO+DAPO loop on the popularity/ELO dimension of LLAMIA-Bench.

```bash
# 1. Build dataset (one-off; already done — data/puzzle_{train,val}.parquet)
python -m data.prepare_puzzles --train-limit 2000 --val-limit 200

# 2. Smoke-test rollouts (needs external vllm on port 7000 + lc0 on 7100)
python -m scripts.test_rollout -n 20

# 3. Stop external vllm (frees ~73 GB/GPU; lc0 stays up)
pkill -f "vllm serve Qwen"

# 4. Toy training (all 8 GPUs shared with lc0; 100 steps smoke, then lift)
./scripts/train_puzzle_grpo.sh trainer.total_training_steps=100

# Smoke-run (5 steps, no val, console-only logging):
./scripts/train_puzzle_grpo.sh \
  trainer.total_training_steps=5 \
  trainer.val_before_train=false \
  'trainer.logger=["console"]' \
  data.train_batch_size=16 \
  'actor_rollout_ref.rollout.n=4'
```

**Reward**: format-gated regression. Answer must match `popularity is <int> and the ELO is <int>`.
Score = `fmt × (½·R_pop + ½·R_elo)`, `R_x = max(0, 1 − |err|/scale)`, scales 50 / 400.
`compute_score` returns a dict so VERL stores `format_pass`, `pop_err`, `elo_err`, `num_tool_calls`, `solution_len` in `batch.non_tensor_batch` alongside the scalar reward.
See `rewards/puzzle_reward.py`.

**Trace logging**: every scored trajectory is appended to `logs/puzzle_traces.jsonl`
(O_APPEND atomic writes; safe across Ray worker processes). Override path with
`PUZZLE_TRACE_LOG=path` env var.

```bash
# Live rolling stats (run in a second terminal alongside training):
python scripts/watch_reward.py --window 64 --interval 5

# Post-hoc analysis (reward mean±std, format rate, tool breakdown, examples):
python scripts/analyze_traces.py logs/puzzle_traces.jsonl --last 500 --examples 3
```

**Tools at training time**: VERL `tool_agent_loop` (`@register("tool_agent")`) drives
multi-turn rollouts via `agents/verl_tool_config.yaml`. Wrappers in `agents/verl_tools.py`
share one `ChessState` per trajectory via `agent_data.extra_fields["chess_board"]`.

**Config**: `configs/qwen3_puzzle_grpo.yaml`. Verified VERL 0.7.1 knobs:
- `actor.loss_agg_mode: token-mean` — token-level policy gradient (DAPO)
- `actor.clip_ratio_low/high: 0.2/0.28` — Clip-Higher (DAPO)
- `algorithm.adv_estimator: grpo` + `rollout.n: 8` — GRPO group size
- `rollout.multi_turn.enable: true` + `agent.default_agent_loop: tool_agent`
- `data.apply_chat_template_kwargs: {enable_thinking: false}` — suppress Qwen3 think block

Not in verl 0.7.1 schema (DAPO-deferred): dynamic_sampling, overlong_buffer penalty.

### When RL goes wrong: rl_recipes.md

`rl_recipes.md` (in repo root) is the practitioner's guide for tool-call GRPO/DAPO failure
modes on Qwen3-class models. **Always consult it before patching training symptoms.**

| Symptom | Section | Fix-of-first-resort |
|---|---|---|
| Reward stuck at 0.0 | §4 format/mode collapse | Verify format-pass on raw rollouts; SFT cold-start if <50% |
| Reward variance → 0 mid-training | §4 echo trap | Confirm dynamic sampling (if available); lower KL β |
| Tool-call rate drops to 0 | §4 direct-answer collapse | Add zero-tool-trajectory penalty; raise temperature |
| Trajectory length explodes | §5 length bias | Token-Level Loss + Overlong soft penalty (deferred, monitor manually) |
| Entropy collapse | §1 Clip-Higher | Verify ε_high=0.28; check ε_low isn't raised |
| Qwen3 plans tool but never emits | §6 thinking-mode quirk | `enable_thinking: false` in rollout config |
