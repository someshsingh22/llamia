# Toy Puzzle-Difficulty RL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run end-to-end VERL RL on Qwen3.5-4B for a single toy task — predicting Lichess puzzle *popularity* (∈ [-100, 100]) and *ELO* (~400–3000) — with the tool-using chess analyst as the rollout policy. Optimize for "fail fast and learn": every task ends in a runnable artifact that surfaces real failure modes early.

**Architecture:**
- Dataset: `ssingh22/llamia-chess-data` config `behavioural_cloning`, filter `class == "type_Puzzle"`. Build a parquet with `prompt` (system + user), `reward_model.ground_truth = {popularity, elo}`, and the FEN as a side-channel. Skip the themes/solution turns from the source conversation.
- Rollout: VERL `tool_agent_loop` (multi-turn, async, vllm-backed). Tools are `analyze`, `get_policy`, `get_position`, `make_move`, `undo_move` — exposed as `verl.tools.BaseTool` subclasses wrapping the existing `agents/tools.py` `LcOClient`.
- Reward: `format × (½·R_pop + ½·R_elo)` where `R_x = max(0, 1 − |err_x|/scale_x)`. Format is binary on parsing the final answer with regex `popularity .* (-?\d+).* ELO .* (\d+)`.
- Algorithm: GRPO with DAPO knobs preset (Clip-Higher ε=(0.2, 0.28), Token-Level Loss, Overlong Soft Penalty, Dynamic Sampling, β_KL=1e-3 with 100-step ref refresh). Group size G=8, T=1.0 rollout. Loss is masked on tool/observation tokens (verl default for tool_agent_loop). `enable_thinking=False` to dodge the Qwen3 "plans tool but never emits" failure mode.
- GPU layout: lc0 on GPUs 0–3 (ports 7100–7103), VERL training on GPUs 4–7. The vllm rollout server stays on its current 8-GPU pool on port 7000 — VERL will use it via the rollout-server pathway, not co-trained on the same GPUs as lc0.

**Tech Stack:** VERL 0.7.1, vLLM 0.14.0, Qwen3.5-4B, lc0 0.32.1 BT4, PyArrow/Parquet, HuggingFace `datasets`, FastAPI lc0 wrapper.

**Failure-mode reference:** When training runs go sideways, consult `rl_recipes.md` §4 (failure-mode table) and §5 (tricks) before tweaking. Keep the symptom→mitigation mapping visible.

---

## File Structure

| Path | Purpose | Status |
|------|---------|--------|
| `data/prepare_puzzles.py` | CLI: load HF dataset, filter Puzzles, parse popularity+ELO, write parquet | new |
| `data/puzzle_train.parquet`, `data/puzzle_val.parquet` | Generated artifacts | generated |
| `rewards/__init__.py` | namespace | new |
| `rewards/puzzle_reward.py` | `compute_score(...)` for VERL custom reward | new |
| `rewards/parser.py` | `parse_popularity_elo(text) -> (pop|None, elo|None)` shared by reward + rollout test | new |
| `tests/test_puzzle_reward.py` | unit tests for parser + reward | new |
| `tests/test_prepare_puzzles.py` | unit test for parquet shape | new |
| `agents/verl_tools.py` | `LcoAnalyzeTool`, `LcoPolicyTool`, `GetPositionTool`, `MakeMoveTool`, `UndoMoveTool` — `verl.tools.BaseTool` subclasses sharing one `ChessState` per trajectory | new |
| `agents/verl_tool_config.yaml` | VERL tool registry pointing to the classes above | new |
| `scripts/test_rollout.py` | smoke test: runs N puzzles through `ChessAnalyst`, parses, scores, prints stats | new |
| `scripts/disaggregate_lc0.sh` | restart lc0 pool on GPUs 0–3 only | new |
| `configs/qwen3_puzzle_grpo.yaml` | VERL config for the toy task (DAPO knobs, custom reward, agent loop, ground-truth columns) | new |
| `scripts/train_puzzle_grpo.sh` | wrapper invoking `verl.trainer.main_ppo` | new |
| `CLAUDE.md` | add "RL Training" section pointing at `rl_recipes.md` + new artifacts | modify |
| `llamiaTex/pages/appendix/details.tex` | mirror config / reward / tool registry into paper | modify |

---

## Pre-flight

- [ ] **Step 0.1: Verify environment is healthy**

```bash
source /dev/shm/somesh/llamia/.venv/bin/activate
python -c "import verl, vllm, datasets, chess, openai, httpx, pyarrow; print('ok', verl.__version__)"
curl -sf http://localhost:7000/v1/models | head -c 200; echo
curl -sf http://localhost:7100/health
```

Expected: `ok 0.7.1`, a JSON response listing `Qwen/Qwen3.5-4B`, `{"status":"ok"}` from lc0.
If any check fails, stop and surface the failure — do not paper over.

- [ ] **Step 0.2: Create branch + commit pre-flight check**

```bash
git checkout -b puzzle-rl-toy
git status
```

No commit yet — just a clean branch.

---

## Task 1: Dataset preparation

**Files:**
- Create: `data/prepare_puzzles.py`
- Test: `tests/test_prepare_puzzles.py`

The HF dataset rows look like:
```python
{
  "conversations": [
    {"from": "human", "value": "Your task is to describe and then solve... <state>\n What are the themes for this puzzle?"},
    {"from": "gpt",   "value": "advantage, endgame, short"},
    {"from": "human", "value": "What is the solution to the puzzle? Answer must be in UCI notation."},
    {"from": "gpt",   "value": "a1a2 b3c1 a2a3 d1b2"},
    {"from": "human", "value": "What would be the popularity and the ELO of the puzzle..."},
    {"from": "gpt",   "value": "The popularity is 88 and the ELO is 1904"},
  ],
  "state": ["6k1/5pp1/7p/8/1p4Pn/1N2P3/1n2KP1P/r2N3R b - - 1 34"],
  "class": "type_Puzzle",
}
```
Only the FEN + the popularity/ELO turn matter for this task.

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_prepare_puzzles.py`:

```python
"""Smoke-test the puzzle parquet builder on a tiny offline fixture."""
from pathlib import Path

import pyarrow.parquet as pq

from data.prepare_puzzles import build_rows, GOLD_RE


def test_gold_regex_extracts_popularity_and_elo():
    pop, elo = GOLD_RE.search("The popularity is 88 and the ELO is 1904").groups()
    assert int(pop) == 88
    assert int(elo) == 1904

    pop, elo = GOLD_RE.search("The popularity is -27 and the ELO is 2350").groups()
    assert int(pop) == -27
    assert int(elo) == 2350


def test_build_rows_extracts_fen_and_targets():
    raw = [{
        "class": "type_Puzzle",
        "state": ["6k1/5pp1/7p/8/1p4Pn/1N2P3/1n2KP1P/r2N3R b - - 1 34"],
        "conversations": [
            {"from": "human", "value": "ignored q1 <state>"},
            {"from": "gpt",   "value": "ignored a1"},
            {"from": "human", "value": "ignored q2"},
            {"from": "gpt",   "value": "ignored a2"},
            {"from": "human", "value": "What would be the popularity and the ELO of the puzzle..."},
            {"from": "gpt",   "value": "The popularity is 88 and the ELO is 1904"},
        ],
    }]

    rows = build_rows(raw)
    assert len(rows) == 1
    r = rows[0]
    assert r["fen"] == "6k1/5pp1/7p/8/1p4Pn/1N2P3/1n2KP1P/r2N3R b - - 1 34"
    assert r["reward_model"]["ground_truth"] == {"popularity": 88, "elo": 1904}
    assert isinstance(r["prompt"], list) and r["prompt"][0]["role"] == "system"
    assert "popularity" in r["prompt"][-1]["content"].lower()
    assert r["fen"] in r["prompt"][-1]["content"]


def test_non_puzzle_rows_are_dropped():
    raw = [{"class": "type_CCRL", "state": ["x"], "conversations": []}]
    assert build_rows(raw) == []
```

- [ ] **Step 1.2: Run the test and watch it fail**

```bash
cd /dev/shm/somesh/llamia && pytest tests/test_prepare_puzzles.py -v
```

Expected: `ImportError: cannot import name 'build_rows' from 'data.prepare_puzzles'`.

- [ ] **Step 1.3: Implement `data/prepare_puzzles.py`**

```python
"""Build the puzzle popularity+ELO parquet for VERL training.

Filters `ssingh22/llamia-chess-data` (config `behavioural_cloning`) for
`class == "type_Puzzle"`, extracts the FEN and the popularity/ELO targets
from the third gpt turn, and writes a parquet whose schema matches what
the VERL data loader expects: a `prompt` column (list[dict]) and a
`reward_model` column with `style`, `ground_truth`, and a side-channel
`fen` for the rollout-time tools.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

GOLD_RE = re.compile(r"popularity is (-?\d+) and the ELO is (\d+)")

SYSTEM_PROMPT = (
    "You are a chess expert with access to lc0, a top neural network engine.\n"
    "A board has been loaded with the puzzle position. Use the engine tools to\n"
    "analyse it, then estimate two numbers:\n"
    "  - popularity: an integer in [-100, 100] (Lichess upvote score; 100 = excellent)\n"
    "  - ELO: an integer roughly in [400, 3000] (puzzle difficulty rating)\n"
    "Tools (state is shared across calls):\n"
    "  - get_position\n"
    "  - analyze(nodes, multipv, moves=[])\n"
    "  - get_policy(nodes)\n"
    "  - make_move(move) / undo_move\n"
    "When two tool calls are independent, emit BOTH in the same response — they run in parallel.\n"
    "When you are done analysing, finish with EXACTLY one line of the form:\n"
    "  The popularity is <int> and the ELO is <int>\n"
    "Do not output anything after that line."
)

USER_TEMPLATE = (
    "FEN: {fen}\n"
    "What would be the popularity and the ELO of this puzzle? "
    "Popularity is between -100 and 100; ELO is the Lichess puzzle rating."
)


def build_rows(raw: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for ex in raw:
        if ex.get("class") != "type_Puzzle":
            continue
        state = ex.get("state") or []
        if not state:
            continue
        fen = state[0]
        gold_text = ""
        for msg in ex.get("conversations", []):
            if msg.get("from") == "gpt" and "popularity" in msg.get("value", "").lower():
                gold_text = msg["value"]
                break
        m = GOLD_RE.search(gold_text)
        if not m:
            continue
        pop, elo = int(m.group(1)), int(m.group(2))
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(fen=fen)},
            ],
            "reward_model": {
                "style": "puzzle_popularity_elo",
                "ground_truth": {"popularity": pop, "elo": elo},
            },
            "fen": fen,
            "data_source": "ssingh22/llamia-chess-data",
        })
    return rows


def _stream_split(split: str, limit: int | None):
    from datasets import load_dataset
    ds = load_dataset("ssingh22/llamia-chess-data", "behavioural_cloning", split=split, streaming=True)
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train-limit", type=int, default=4000)
    ap.add_argument("--val-limit", type=int, default=400)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, limit, name in [
        ("train", args.train_limit, "puzzle_train.parquet"),
        ("test", args.val_limit, "puzzle_val.parquet"),
    ]:
        rows = build_rows(_stream_split(split, limit * 5))  # 5x headroom for filter
        rows = rows[:limit]
        table = pa.Table.from_pylist(rows)
        out = args.out_dir / name
        pq.write_table(table, out)
        print(f"{name}: {len(rows)} puzzles → {out}")


if __name__ == "__main__":
    main()
```

Also create `data/__init__.py` (empty) so imports work:

```bash
touch /dev/shm/somesh/llamia/data/__init__.py
```

- [ ] **Step 1.4: Re-run the test and watch it pass**

```bash
cd /dev/shm/somesh/llamia && pytest tests/test_prepare_puzzles.py -v
```

Expected: 3 passing.

- [ ] **Step 1.5: Build the actual parquet and eyeball it**

```bash
cd /dev/shm/somesh/llamia && python -m data.prepare_puzzles --train-limit 2000 --val-limit 200
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/puzzle_train.parquet')
print('rows:', t.num_rows, 'cols:', t.column_names)
r = t.to_pylist()[0]
print('FEN:', r['fen'])
print('GT :', r['reward_model']['ground_truth'])
print('USR:', r['prompt'][1]['content'][:140])
"
```

Expected: ~2000 rows; FEN looks like a real position; ground truth has integer `popularity` and `elo`.

- [ ] **Step 1.6: Commit**

```bash
git add data/prepare_puzzles.py data/__init__.py tests/test_prepare_puzzles.py
git add data/puzzle_train.parquet data/puzzle_val.parquet 2>/dev/null || true
git commit -m "data: build puzzle popularity+ELO parquet from llamia-chess-data"
```

(Parquet files may be too large to commit; if so, .gitignore them and commit only code. Decide based on `git status` output before committing.)

---

## Task 2: Reward function (TDD)

**Files:**
- Create: `rewards/__init__.py`, `rewards/parser.py`, `rewards/puzzle_reward.py`
- Test: `tests/test_puzzle_reward.py`

Reward design (multiplicative, per `rl_recipes.md` §2):

```
parsed_pop, parsed_elo = parse(final_answer)
fmt = 1.0 if (parsed_pop is not None and parsed_elo is not None) else 0.0
R_pop = max(0.0, 1.0 - abs(parsed_pop - gold_pop) / 50.0)   # 50 cp on a -100..100 scale
R_elo = max(0.0, 1.0 - abs(parsed_elo - gold_elo) / 400.0)  # 400 Elo on ~400..3000 scale
score = fmt * (0.5 * R_pop + 0.5 * R_elo)
```

Notes:
- Multiplicative composition: a bad format zeros the outcome reward. Avoids "answer-shaped scribbles" gaming format-only rewards (rl_recipes §2 hacking patterns).
- Read-only tool calls earn **zero** reward (we simply don't add per-tool bonuses) — per IRC, positive per-call rewards drive over-invocation.
- No format bonus separate from the gate. We're not adding +0.1 for valid format because format is already a hard gate on the outcome score.

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_puzzle_reward.py`:

```python
import math

from rewards.parser import parse_popularity_elo
from rewards.puzzle_reward import compute_score


def test_parser_canonical_form():
    assert parse_popularity_elo("The popularity is 88 and the ELO is 1904") == (88, 1904)


def test_parser_handles_negative_popularity():
    assert parse_popularity_elo("popularity is -23 and the ELO is 2100") == (-23, 2100)


def test_parser_picks_last_match_if_model_rambles():
    text = "earlier I guessed popularity is 50 and the ELO is 1000.\nFinal: The popularity is 88 and the ELO is 1904"
    assert parse_popularity_elo(text) == (88, 1904)


def test_parser_returns_none_on_garbage():
    assert parse_popularity_elo("I don't know") == (None, None)


def test_score_perfect_prediction():
    s = compute_score(
        data_source="ssingh22/llamia-chess-data",
        solution_str="The popularity is 88 and the ELO is 1904",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert math.isclose(s, 1.0)


def test_score_format_failure_zeros_reward():
    s = compute_score(
        data_source="x",
        solution_str="I think it's a hard puzzle.",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert s == 0.0


def test_score_partial_credit_on_close_predictions():
    # 25 popularity off, 200 ELO off → 0.5 * 0.5 + 0.5 * 0.5 = 0.5
    s = compute_score(
        data_source="x",
        solution_str="The popularity is 63 and the ELO is 1704",
        ground_truth={"popularity": 88, "elo": 1904},
        extra_info=None,
    )
    assert math.isclose(s, 0.5, rel_tol=1e-6)


def test_score_clamps_far_misses_to_zero():
    s = compute_score(
        data_source="x",
        solution_str="The popularity is -100 and the ELO is 400",
        ground_truth={"popularity": 100, "elo": 3000},
        extra_info=None,
    )
    assert s == 0.0
```

- [ ] **Step 2.2: Run tests, watch them fail**

```bash
cd /dev/shm/somesh/llamia && pytest tests/test_puzzle_reward.py -v
```

Expected: ImportError on `rewards.parser` / `rewards.puzzle_reward`.

- [ ] **Step 2.3: Implement parser**

Create `rewards/__init__.py` (empty) and `rewards/parser.py`:

```python
"""Parse 'popularity is X and the ELO is Y' from rollout completions.

Tolerant: matches anywhere in the string; if multiple matches exist,
returns the LAST one (model often shows scratch work then concludes).
"""
from __future__ import annotations

import re

_PATTERN = re.compile(r"popularity\s+is\s+(-?\d+)\s+and\s+the\s+ELO\s+is\s+(\d+)", re.IGNORECASE)


def parse_popularity_elo(text: str) -> tuple[int | None, int | None]:
    matches = list(_PATTERN.finditer(text or ""))
    if not matches:
        return (None, None)
    m = matches[-1]
    return int(m.group(1)), int(m.group(2))
```

- [ ] **Step 2.4: Implement reward**

Create `rewards/puzzle_reward.py`:

```python
"""VERL custom reward for the puzzle popularity+ELO task.

Function signature matches `verl.trainer.config.RewardModelConfig`'s
`custom_reward_function` contract (data_source, solution_str, ground_truth, extra_info).
"""
from __future__ import annotations

from typing import Any

from .parser import parse_popularity_elo

POP_SCALE = 50.0   # within ±50 popularity points → linear partial credit
ELO_SCALE = 400.0  # within ±400 Elo → linear partial credit


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, int],
    extra_info: dict[str, Any] | None = None,
) -> float:
    pop_pred, elo_pred = parse_popularity_elo(solution_str)
    if pop_pred is None or elo_pred is None:
        return 0.0  # format gate → multiplicative zero
    pop_err = abs(pop_pred - int(ground_truth["popularity"]))
    elo_err = abs(elo_pred - int(ground_truth["elo"]))
    r_pop = max(0.0, 1.0 - pop_err / POP_SCALE)
    r_elo = max(0.0, 1.0 - elo_err / ELO_SCALE)
    return 0.5 * r_pop + 0.5 * r_elo
```

- [ ] **Step 2.5: Re-run, watch them pass**

```bash
cd /dev/shm/somesh/llamia && pytest tests/test_puzzle_reward.py -v
```

Expected: 7 passing.

- [ ] **Step 2.6: Commit**

```bash
git add rewards/ tests/test_puzzle_reward.py
git commit -m "rewards: format-gated popularity+ELO regression reward"
```

---

## Task 3: Rollout smoke test (Fail-Fast Checkpoint #1)

**Goal:** Before touching VERL, confirm Qwen3.5-4B *as currently served* can run the chess analyst on real puzzles, emit the required answer line, and that our reward pipeline scores it sensibly. If this fails, we know to add an SFT cold-start before RL (rl_recipes §3).

**Files:**
- Create: `scripts/test_rollout.py`

- [ ] **Step 3.1: Write the smoke-test script**

Create `scripts/test_rollout.py`:

```python
"""Run N puzzles through ChessAnalyst, parse the final answer, score with
puzzle reward, and print aggregate stats.

Fail-fast diagnostic — NOT a unit test. Output is human-readable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq

from agents.chess_analyst import ChessAnalyst
from rewards.parser import parse_popularity_elo
from rewards.puzzle_reward import compute_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/puzzle_val.parquet"))
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--llm-url", default="http://localhost:7000/v1")
    ap.add_argument("--lc0-urls", nargs="+", default=None)
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()[: args.n]
    print(f"Running {len(rows)} puzzles…\n")

    scores: list[float] = []
    fmt_ok = 0
    tool_calls_per_run: list[int] = []
    rounds_per_run: list[int] = []
    t0 = time.time()

    for i, row in enumerate(rows):
        fen = row["fen"]
        gt = row["reward_model"]["ground_truth"]
        analyst = ChessAnalyst(
            llm_base_url=args.llm_url,
            lc0_urls=args.lc0_urls,
            initial_fen=fen,
        )
        # Build the user prompt the same way the dataset does — use the prompt list directly.
        # ChessAnalyst.run takes a query string; reuse the user content.
        user_content = row["prompt"][-1]["content"]
        try:
            answer = analyst.run(user_content, verbose=False)
        except Exception as e:
            print(f"[{i:3d}] ROLLOUT ERROR: {type(e).__name__}: {e}")
            scores.append(0.0)
            continue

        pop, elo = parse_popularity_elo(answer)
        score = compute_score(
            data_source="x", solution_str=answer,
            ground_truth=gt, extra_info=None,
        )
        scores.append(score)
        if pop is not None:
            fmt_ok += 1
        stats = analyst.last_stats
        tool_calls_per_run.append(stats.get("tool_calls", 0))
        rounds_per_run.append(stats.get("llm_rounds", 0))

        print(
            f"[{i:3d}] gt=(pop={gt['popularity']}, elo={gt['elo']}) "
            f"pred=(pop={pop}, elo={elo}) score={score:.3f} "
            f"tools={stats.get('tool_calls')} rounds={stats.get('llm_rounds')}"
        )

    elapsed = time.time() - t0
    print("\n=== Rollout smoke-test summary ===")
    print(f"  N                : {len(scores)}")
    print(f"  Wall time        : {elapsed:.1f}s ({elapsed/len(scores):.1f}s/puzzle)")
    print(f"  Format-pass rate : {fmt_ok/len(scores):.1%}")
    print(f"  Mean reward      : {statistics.mean(scores):.3f}")
    print(f"  Reward stdev     : {statistics.pstdev(scores):.3f}")
    if tool_calls_per_run:
        print(f"  Tool calls       : mean={statistics.mean(tool_calls_per_run):.1f} "
              f"max={max(tool_calls_per_run)}")
        print(f"  LLM rounds       : mean={statistics.mean(rounds_per_run):.1f} "
              f"max={max(rounds_per_run)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run the smoke test on 20 puzzles**

```bash
cd /dev/shm/somesh/llamia && python -m scripts.test_rollout -n 20 2>&1 | tee /tmp/rollout_smoke.log
```

**Decision point — read carefully:**
- **Format-pass rate ≥ 50% AND mean reward ≥ 0.15**: green light, proceed to Task 4.
- **Format-pass rate < 50%**: the model isn't producing the answer line. Tighten `SYSTEM_PROMPT` in `data/prepare_puzzles.py` (e.g., add a one-shot example), regenerate parquets, re-run smoke. If still <50% after one prompt iteration, STOP and reassess — we likely need SFT cold-start before RL (rl_recipes §3, "Why cold start is uniquely bad for tool-call tasks"). Do NOT proceed to VERL training without this.
- **Format ≥ 50% but mean reward = 0.0**: parser bug or off-by-one. Inspect raw answers in the log.
- **Tool-calls mean = 0**: model is answering without calling tools. Acceptable for now (toy task) but note it — at training time this surfaces as the "mode collapse to direct answer" failure (rl_recipes §4).

- [ ] **Step 3.3: Commit the script + log a short verdict**

Save the smoke log in the commit message:

```bash
cd /dev/shm/somesh/llamia
SUMMARY=$(tail -10 /tmp/rollout_smoke.log)
git add scripts/test_rollout.py
git commit -m "scripts: rollout smoke test for puzzle popularity+ELO

Initial run on 20 val puzzles:
$SUMMARY"
```

---

## Task 4: GPU disaggregation

**Files:**
- Create: `scripts/disaggregate_lc0.sh`

Per `rl_recipes.md` §7: do not co-locate inference and training on the same GPUs at scale. lc0 currently runs on GPUs 0–7 (8-server pool), and VERL training will also want all 8. We move lc0 to GPUs 0–3 (ports 7100–7103) and reserve GPUs 4–7 for VERL.

- [ ] **Step 4.1: Write the disaggregation script**

Create `scripts/disaggregate_lc0.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Restart lc0 pool on GPUs 0-3 only (ports 7100-7103),
# leaving GPUs 4-7 free for VERL training.
cd "$(dirname "$0")/.."

# Kill existing lc0 servers
pkill -f "lc0_server" || true
sleep 2

# Start 4 servers on GPUs 0-3
N_GPUS=4 BASE_PORT=7100 LOG_DIR=/tmp \
  ./lc0_server/scripts/start_lc0_servers.sh

# Health-check
for p in 7100 7101 7102 7103; do
  for _ in {1..30}; do
    if curl -sf "http://localhost:$p/health" >/dev/null; then
      echo "  port $p OK"
      break
    fi
    sleep 1
  done
done
echo "lc0 disaggregated to GPUs 0-3 (ports 7100-7103)"
```

- [ ] **Step 4.2: Run it and verify**

```bash
chmod +x /dev/shm/somesh/llamia/scripts/disaggregate_lc0.sh
/dev/shm/somesh/llamia/scripts/disaggregate_lc0.sh
nvidia-smi --query-gpu=index,memory.used --format=csv | head -10
```

Expected: GPUs 0–3 show ~3 GB used (lc0 + vllm shard); GPUs 4–7 show vllm shard only. All 4 health checks pass.

- [ ] **Step 4.3: Re-run the rollout smoke test against the 4-server pool**

```bash
cd /dev/shm/somesh/llamia && python -m scripts.test_rollout -n 10 \
  --lc0-urls http://localhost:7100 http://localhost:7101 \
             http://localhost:7102 http://localhost:7103
```

Expected: comparable format-pass rate and mean reward as Task 3.2. Wall time may rise ~2× (half the lc0 capacity); acceptable.

- [ ] **Step 4.4: Update `agents/chess_analyst.py` default pool**

Edit `agents/chess_analyst.py` line 188:

Before:
```python
_DEFAULT_LC0_POOL = [f"http://localhost:{7100 + i}" for i in range(8)]
```

After:
```python
_DEFAULT_LC0_POOL = [f"http://localhost:{7100 + i}" for i in range(4)]
```

- [ ] **Step 4.5: Commit**

```bash
cd /dev/shm/somesh/llamia
git add scripts/disaggregate_lc0.sh agents/chess_analyst.py
git commit -m "infra: run lc0 on GPUs 0-3 only; reserve 4-7 for VERL training"
```

---

## Task 5: VERL tool wrappers

**Files:**
- Create: `agents/verl_tools.py`
- Create: `agents/verl_tool_config.yaml`

VERL `tool_agent_loop` calls `BaseTool.create(instance_id, **kw)` once per trajectory and `BaseTool.execute(instance_id, parameters, **kw)` per call. State must be carried per-instance, so each tool keeps a `dict[instance_id -> ChessState]`. The FEN comes in via `kw["fen"]` (passed by VERL from the dataset's `extra_info` — we wire that in Task 6).

- [ ] **Step 5.1: Implement the tool wrappers**

Create `agents/verl_tools.py`:

```python
"""VERL BaseTool wrappers around the chess analyst's tool surface.

One ChessState per trajectory (keyed by instance_id). The lc0 client is
shared module-wide and round-robins across the lc0 pool.
"""
from __future__ import annotations

from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .state import ChessState
from .tools import LcOClient

import chess
import json

_DEFAULT_LC0_POOL = [f"http://localhost:{7100 + i}" for i in range(4)]
_LC0 = LcOClient(base_urls=_DEFAULT_LC0_POOL)


class _StatefulTool(BaseTool):
    """Shared per-trajectory ChessState bookkeeping."""

    _states: dict[str, ChessState] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        instance_id = instance_id or kwargs.get("instance_id") or ""
        fen = kwargs.get("fen") or chess.STARTING_FEN
        self._states[instance_id] = ChessState(fen)
        return instance_id, ToolResponse(text=f"Board loaded with FEN {fen}.")

    async def release(self, instance_id: str, **kwargs) -> None:
        self._states.pop(instance_id, None)


class GetPositionTool(_StatefulTool):
    async def execute(self, instance_id, parameters, **kwargs):
        info = self._states[instance_id].info()
        return ToolResponse(text=json.dumps(info)), 0.0, {}


class MakeMoveTool(_StatefulTool):
    async def execute(self, instance_id, parameters, **kwargs):
        r = self._states[instance_id].make_move(parameters.get("move", ""))
        return ToolResponse(text=json.dumps(r)), 0.0, {}


class UndoMoveTool(_StatefulTool):
    async def execute(self, instance_id, parameters, **kwargs):
        r = self._states[instance_id].undo_move()
        return ToolResponse(text=json.dumps(r)), 0.0, {}


class AnalyzeTool(_StatefulTool):
    async def execute(self, instance_id, parameters, **kwargs):
        st = self._states[instance_id]
        r = _LC0.analyze(
            st.fen,
            nodes=int(parameters.get("nodes", 800)),
            multipv=int(parameters.get("multipv", 3)),
            moves=parameters.get("moves"),
        )
        return ToolResponse(text=json.dumps(r)), 0.0, {}


class GetPolicyTool(_StatefulTool):
    async def execute(self, instance_id, parameters, **kwargs):
        st = self._states[instance_id]
        nodes = parameters.get("nodes")
        r = _LC0.get_policy(
            st.fen,
            nodes=int(nodes) if nodes is not None else None,
        )
        return ToolResponse(text=json.dumps(r)), 0.0, {}
```

- [ ] **Step 5.2: Write the tool registry config**

Create `agents/verl_tool_config.yaml` (each entry mirrors a function in `agents/chess_analyst.py:TOOL_DEFS`):

```yaml
tools:
  - class_name: agents.verl_tools.GetPositionTool
    config: {}
    tool_schema:
      type: function
      function:
        name: get_position
        description: Return current FEN, side to move, legal moves, and recent move history.
        parameters: {type: object, properties: {}, required: []}

  - class_name: agents.verl_tools.AnalyzeTool
    config: {}
    tool_schema:
      type: function
      function:
        name: analyze
        description: lc0 search; pass moves=[X] to evaluate AFTER X without changing state.
        parameters:
          type: object
          properties:
            nodes:   {type: integer}
            multipv: {type: integer}
            moves:   {type: array, items: {type: string}}
          required: []

  - class_name: agents.verl_tools.GetPolicyTool
    config: {}
    tool_schema:
      type: function
      function:
        name: get_policy
        description: lc0 raw NN priors P and per-move values V.
        parameters:
          type: object
          properties:
            nodes: {type: integer}
          required: []

  - class_name: agents.verl_tools.MakeMoveTool
    config: {}
    tool_schema:
      type: function
      function:
        name: make_move
        description: Apply a UCI or SAN move to the current board.
        parameters:
          type: object
          properties:
            move: {type: string}
          required: [move]

  - class_name: agents.verl_tools.UndoMoveTool
    config: {}
    tool_schema:
      type: function
      function:
        name: undo_move
        description: Undo the last move.
        parameters: {type: object, properties: {}, required: []}
```

- [ ] **Step 5.3: Smoke-test the tools standalone**

```bash
cd /dev/shm/somesh/llamia && python -c "
import asyncio, json
from agents.verl_tools import AnalyzeTool, GetPositionTool
from verl.tools.schemas import OpenAIFunctionToolSchema

schema = OpenAIFunctionToolSchema.model_validate({
  'type': 'function',
  'function': {'name': 'analyze', 'description': 'x',
               'parameters': {'type': 'object', 'properties': {}, 'required': []}},
})

async def main():
    t = AnalyzeTool(config={}, tool_schema=schema)
    iid, _ = await t.create(instance_id='test1', fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    resp, score, _ = await t.execute(iid, {'nodes': 200, 'multipv': 1})
    print('score:', score)
    print('resp:', resp.text[:200])
    await t.release(iid)

asyncio.run(main())
"
```

Expected: a JSON string with `bestmove` and a `multipv` array; no exceptions.

- [ ] **Step 5.4: Commit**

```bash
git add agents/verl_tools.py agents/verl_tool_config.yaml
git commit -m "agents: VERL BaseTool wrappers around lc0 chess tools"
```

---

## Task 6: VERL training config (Fail-Fast Checkpoint #2)

**Files:**
- Modify: `configs/qwen3_ppo_vllm.yaml` (deprecate — keep but unused)
- Create: `configs/qwen3_puzzle_grpo.yaml`
- Create: `scripts/train_puzzle_grpo.sh`

Knobs are set per `rl_recipes.md` §1 (DAPO defaults for dense Qwen3) and §5 (defaults for 7B-class models — we scale slightly down for 4B).

- [ ] **Step 6.1: Write the training config**

Create `configs/qwen3_puzzle_grpo.yaml`:

```yaml
defaults:
  - ppo_trainer
  - _self_

# ── Model: actual served model (Qwen3.5-4B), not the 122B placeholder ──
actor_rollout_ref:
  model:
    path: Qwen/Qwen3.5-4B
    use_remove_padding: True
    enable_gradient_checkpointing: True
    trust_remote_code: True

  actor:
    optim:
      lr: 1e-6                       # rl_recipes §5: 1e-6 cosine for 7B-class Qwen3; 4B same range
      lr_warmup_steps_ratio: 0.05    # 5% warmup
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 1
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 8192
    use_token_level_loss: True       # DAPO Token-Level Policy Gradient Loss
    clip_ratio_low: 0.2              # DAPO Clip-Higher
    clip_ratio_high: 0.28
    entropy_coeff: 0.0               # entropy is preserved by Clip-Higher; no extra bonus
    fsdp_config:
      param_offload: False
      optimizer_offload: False
      fsdp_size: -1

  rollout:
    name: vllm
    mode: async
    nnodes: 1
    n_gpus_per_node: 4               # GPUs 4-7 (set via CUDA_VISIBLE_DEVICES in the launch script)
    temperature: 1.0                 # rl_recipes §5: T=1.0 early; anneal later
    top_p: 0.95
    top_k: 20
    response_length: 4096
    prompt_length: 1024
    max_model_len: 8000
    max_num_batched_tokens: 16384
    gpu_memory_utilization: 0.85
    enable_prefix_caching: True
    tensor_model_parallel_size: 1
    dtype: bfloat16
    engine_kwargs:
      vllm:
        reasoning_parser: qwen3
        tool_call_parser: hermes
        enable_auto_tool_choice: True
    # ── Multi-turn agent loop with our lc0 tools ──
    agent:
      agent_loop_name: tool_agent
      tool_config_path: agents/verl_tool_config.yaml
      max_turns: 8                   # cap per rollout (chess_analyst uses 14; tighter for toy)
    chat_template_kwargs:
      enable_thinking: False         # rl_recipes §6: dodges the Qwen3 plans-but-no-tool failure

  ref:
    fsdp_config:
      param_offload: True

# ── Critic disabled: GRPO is critic-free ──
critic:
  enable: False

# ── Custom reward ──
reward_model:
  enable: False
custom_reward_function:
  path: rewards/puzzle_reward.py
  name: compute_score

# ── Data ──
data:
  tokenizer: Qwen/Qwen3.5-4B
  train_files: data/puzzle_train.parquet
  val_files: data/puzzle_val.parquet
  prompt_key: prompt
  reward_fn_key: data_source
  max_prompt_length: 1024
  max_response_length: 4096
  train_batch_size: 64               # 8 prompts × G=8 = 64
  val_batch_size: 64
  # extra_info keys forwarded to tool create() — VERL passes the row's `fen`
  # into tool kwargs when it appears here.
  extra_info_keys: [fen]

# ── Algorithm: GRPO + DAPO knobs ──
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: grpo
  group_size: 8                      # G=8: rl_recipes §5 sweet spot for tool-call multi-turn
  use_dynamic_sampling: True         # DAPO: drop groups with zero reward variance
  overlong_buffer_len: 512           # DAPO Overlong soft penalty: linear in last 512 tokens
  overlong_buffer_penalty: 1.0
  kl_ctrl:
    type: fixed
    kl_coef: 0.001                   # small β with periodic ref refresh

# ── Trainer ──
trainer:
  total_epochs: 1                    # toy run
  total_training_steps: 100          # smoke-test cap; lift after first green run
  project_name: llamia
  experiment_name: puzzle_grpo_toy
  logger: ["console", "wandb"]
  n_gpus_per_node: 4
  nnodes: 1
  val_before_train: True
  val_freq: 25
  save_freq: 50
  ref_update_freq: 100               # ref refresh — rl_recipes §5
  default_local_dir: checkpoints/puzzle_grpo_toy
```

(Some knob names — `use_token_level_loss`, `use_dynamic_sampling`, `overlong_buffer_*`, `group_size`, `ref_update_freq`, `extra_info_keys` — depend on the exact verl 0.7.1 schema. If a knob is unknown to verl, the launcher will fail loudly with a config error; fix it then by either renaming to the closest match in `verl/trainer/config/ppo_trainer.yaml` or removing the knob if not yet supported. **Do not silently drop on a typo — verify against `_generated_ppo_trainer.yaml` in verl's site-packages.**)

- [ ] **Step 6.2: Verify config knob names against verl's schema**

```bash
cd /dev/shm/somesh/llamia
VERL_CFG=/dev/shm/somesh/llamia/.venv/lib/python3.10/site-packages/verl/trainer/config/_generated_ppo_trainer.yaml
for k in use_token_level_loss use_dynamic_sampling overlong_buffer_len group_size ref_update_freq agent_loop_name tool_config_path; do
  echo -n "$k: "
  grep -c "$k" "$VERL_CFG" || true
done
```

For any knob with count `0`, search the codebase for the actual name:

```bash
grep -rn "token.level.loss\|dynamic_sampling\|overlong\|group_size\|ref_update" \
  /dev/shm/somesh/llamia/.venv/lib/python3.10/site-packages/verl/trainer \
  | head -30
```

Patch `configs/qwen3_puzzle_grpo.yaml` to use the names that exist. Do this *before* the smoke run so the first crash is a real one, not a config typo.

- [ ] **Step 6.3: Write the launch script**

Create `scripts/train_puzzle_grpo.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Pin training to GPUs 4-7 (lc0 owns 0-3 after disaggregate_lc0.sh)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export VLLM_HOST="${VLLM_HOST:-localhost}"
export VLLM_PORT="${VLLM_PORT:-7000}"

source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m verl.trainer.main_ppo \
  --config-path "$PWD/configs" \
  --config-name qwen3_puzzle_grpo \
  trainer.n_gpus_per_node=4 \
  "$@"
```

- [ ] **Step 6.4: Smoke-run for 5 steps**

```bash
chmod +x /dev/shm/somesh/llamia/scripts/train_puzzle_grpo.sh
cd /dev/shm/somesh/llamia
./scripts/train_puzzle_grpo.sh \
  trainer.total_training_steps=5 \
  trainer.val_before_train=False \
  trainer.logger='["console"]' \
  data.train_batch_size=16 \
  algorithm.group_size=4 \
  2>&1 | tee /tmp/puzzle_grpo_smoke.log
```

**Decision point — read carefully:**
- **5 steps complete, reward mean printed, no OOM**: green light. The plumbing works. Proceed to Task 7.
- **OOM**: drop `gpu_memory_utilization` to 0.7 and `response_length` to 2048 first; if still OOM, drop `train_batch_size` to 8.
- **Config error on a DAPO knob**: revisit Step 6.2; either rename to the verl 0.7.1 equivalent or remove the unsupported knob and note it in CLAUDE.md as deferred.
- **Reward mean exactly 0.0 every step**: format collapse. Open `/tmp/puzzle_grpo_smoke.log`, find a sample rollout, verify the parser. If the model never emits the answer line under VERL's chat template (different from the standalone analyst), this is the Qwen3 thinking-mode "plans tool but never emits" failure (rl_recipes §4) — confirm `enable_thinking: False` is actually being applied in the rollout config.
- **Rollout step hangs > 5 min**: lc0 tool deadlock. Check `nvidia-smi` for GPUs 0–3; tail `/tmp/lc0_*.log`.

- [ ] **Step 6.5: Commit**

```bash
git add configs/qwen3_puzzle_grpo.yaml scripts/train_puzzle_grpo.sh
git commit -m "configs: GRPO+DAPO toy training for puzzle popularity+ELO

Per rl_recipes.md: Clip-Higher (0.2, 0.28), Token-Level Loss, Overlong
penalty, Dynamic Sampling, group_size=8, T=1.0, β_KL=1e-3 + ref refresh.
GPUs 4-7 for training; lc0 disaggregated to 0-3."
```

---

## Task 7: CLAUDE.md + details.tex updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `llamiaTex/pages/appendix/details.tex`

- [ ] **Step 7.1: Append to CLAUDE.md**

Add a new section to `CLAUDE.md` (after the existing "RL Training (VERL)" section):

```markdown
### Toy task: puzzle popularity + ELO

End-to-end GRPO+DAPO loop on a single LLAMIA-Bench dimension. Pipeline:

```bash
# 1. Build dataset (one-off)
python -m data.prepare_puzzles --train-limit 4000 --val-limit 400

# 2. Disaggregate lc0 onto GPUs 0-3 (one-off per session)
./scripts/disaggregate_lc0.sh

# 3. Smoke-test rollouts (sanity check; format-pass rate ≥ 50% expected)
python -m scripts.test_rollout -n 20

# 4. Toy training run on GPUs 4-7
./scripts/train_puzzle_grpo.sh trainer.total_training_steps=100
```

**Reward**: format-gated regression. Final answer must match
`popularity is <int> and the ELO is <int>`; reward is
`fmt × (½·R_pop + ½·R_elo)` where `R_x = max(0, 1 − |err|/scale)`,
scales 50 (popularity) and 400 (Elo). See `rewards/puzzle_reward.py`.

**Tools at training time**: VERL's `tool_agent_loop` drives multi-turn
rollouts using `agents/verl_tool_config.yaml`. Tool wrappers live in
`agents/verl_tools.py` and share one `ChessState` per trajectory.

### When RL goes wrong: rl_recipes.md

`rl_recipes.md` (in repo root) is the practitioner's guide for tool-call
GRPO/DAPO failure modes on Qwen3-class models. **Always consult it before
patching training symptoms.** Quick mapping:

| Symptom | Section | Fix-of-first-resort |
|---|---|---|
| Reward stuck at 0.0 | §4 mode/format collapse, §3 cold start | Verify format-pass rate; consider SFT cold-start if <50% |
| Reward variance → 0 mid-training | §4 echo trap | Confirm Dynamic Sampling is on; lower KL β |
| Tool-call rate drops to 0 | §4 mode collapse to direct answer | Add zero-tool-trajectory penalty (IRC); raise temperature |
| Trajectory length explodes | §5 length bias | Ensure Token-Level Loss + Overlong soft penalty are enabled |
| Entropy collapse | §1 Clip-Higher | Verify ε_high=0.28; check ε_low isn't also raised |
| Qwen3 plans tool but never emits | §6 thinking-mode quirk | `enable_thinking=False` in rollout config |
```

- [ ] **Step 7.2: Sync `details.tex`**

Open `llamiaTex/pages/appendix/details.tex` and add (or update) the RL section with: model = Qwen3.5-4B; algorithm = GRPO with DAPO knobs (Clip-Higher 0.2/0.28, Token-Level Loss, Overlong soft penalty, Dynamic Sampling, β_KL=1e-3 with 100-step ref refresh); group size 8; temperature 1.0 for rollouts; max 8 tool-use turns per rollout; reward = `fmt × (½·R_pop + ½·R_elo)` with scales 50 / 400; loss masked on tool/observation tokens; lc0 on GPUs 0–3, training on GPUs 4–7. Match the exact knob names in `configs/qwen3_puzzle_grpo.yaml`.

- [ ] **Step 7.3: Commit**

```bash
cd /dev/shm/somesh/llamia
git add CLAUDE.md
(cd llamiaTex && git add pages/appendix/details.tex && \
  git commit -m "details: add toy puzzle GRPO config + reward")
git add llamiaTex
git commit -m "docs: document toy puzzle GRPO pipeline + rl_recipes.md cross-ref"
```

---

## Self-Review Notes

Done after the plan was drafted:

1. **Spec coverage** — every piece of the user's request is covered: dataset prep (Task 1), prompt design (Task 1's `SYSTEM_PROMPT` / `USER_TEMPLATE`), candidate-rollout test (Task 3), training setup (Task 6), logging (`logger: ["console", "wandb"]` in config + console-only smoke flag).
2. **No placeholders** — every step contains code, commands, or a specific decision. Step 6.2 deliberately includes a "verify before assuming" check rather than asserting verl's exact knob names; this is honest, not a placeholder.
3. **Best-practices baked in** (per user request to derive from `rl_recipes.md`):
   - DAPO knob preset (Clip-Higher 0.2/0.28, Token-Level Loss, Overlong soft penalty, Dynamic Sampling, β_KL=1e-3 + ref refresh).
   - Multiplicative format×outcome reward, no per-call tool bonuses (avoids stuffing).
   - GPU disaggregation (lc0 0–3, training 4–7).
   - `enable_thinking=False` to dodge the Qwen3 plans-but-no-tool quirk.
   - T=1.0 rollout, group size 8, max 8 turns, max 4096 response tokens.
   - Format-pass-rate gate before committing to RL (Task 3 decision point) — explicit cold-start escape hatch.
   - Failure-mode → fix mapping in CLAUDE.md keyed to `rl_recipes.md` sections.
4. **Type consistency** — `compute_score(data_source, solution_str, ground_truth, extra_info)` signature matches in tests, implementation, and config reference. `ground_truth` is `dict[str, int]` with keys `popularity`/`elo` everywhere.
5. **Scope** — single sub-project (one LLAMIA-Bench dimension, popularity+ELO only). Themes/solution turns deliberately skipped per user instruction.
