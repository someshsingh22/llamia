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
