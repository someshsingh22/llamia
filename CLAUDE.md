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

**Run:**
```bash
source .venv/bin/activate
python scripts/run_analyst.py "Why is Nxd4 a blunder here, rnbqkb1r/pp2pppp/5n2/6B1/2pp4/4PN2/PP3PPP/RN1QKB1R w KQkq - 0 6"
```

FENs embedded anywhere in the query are extracted automatically and set as the starting position. Moves accept both UCI (e2e4) and SAN (Nxd4).

**Tool-call format quirk — important:**
Qwen3.5's chat template generates tool calls in its own XML format:
```
<tool_call>
<function=name>
<parameter=key>value</parameter>
</function>
</tool_call>
```
vllm's `--tool-call-parser hermes` finds the `<tool_call>` tags but cannot parse the inner XML as JSON, so `msg.tool_calls` is always `[]`. The agent parses tool calls from `msg.content` with `_parse_qwen_tool_calls()` in `chess_analyst.py`. **Do not rely on the standard `msg.tool_calls` field with this model/server.** Tool results are sent back as standard `role=tool` messages (the Qwen3.5 chat template handles them correctly on the server side).

## RL Training (VERL)

Config: `configs/qwen3_ppo_vllm.yaml`

```bash
VLLM_HOST=localhost VLLM_PORT=7000 ./scripts/train_ppo.sh
# or with overrides:
python -m verl.trainer.main_ppo \
  --config-path configs --config-name qwen3_ppo_vllm \
  data.train_files=data/train.parquet
```
