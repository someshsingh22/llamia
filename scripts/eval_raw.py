#!/usr/bin/env python
"""Raw-output evaluation across models and reasoning efforts.

Edit EVAL_CONFIG below to select which (model, effort) pairs to run.
Each entry produces a separate JSONL in --out-dir with full per-turn traces
including thinking/reasoning tokens, all tool call inputs + outputs, and the
final answer text.

Usage:
    python scripts/eval_raw.py                      # run all entries in EVAL_CONFIG
    python scripts/eval_raw.py --n 10               # seeded sample from processed test split
    python scripts/eval_raw.py --labels gpt-5-low   # run specific label(s)
    python scripts/eval_raw.py --workers 8          # 8 workers → 8 lc0 servers on GPUs 0-7
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from agents.chess_analyst import ChessAnalyst
from agents.metrics import compute_metrics, print_metrics, comparison_table
from data.prepare_puzzles import DEFAULT_SAMPLE_SEED, load_processed_split, sample_rows

# ── Evaluation configuration ──────────────────────────────────────────────────
# Edit this dict to add / remove (model, effort) combinations.
# Keys used:
#   label    - unique run name (used in filename and output)
#   provider - "azure" | "anthropic" | "openai" | "vllm"
#   model    - model identifier string
#   effort   - "low" | "medium" | "high" | None (None = no explicit reasoning control)
EVAL_CONFIG: list[dict] = [
    {"label": "claude-sonnet-4-6-low",    "provider": "azure", "model": "claude-sonnet-4-6",    "effort": "low"},
    {"label": "claude-sonnet-4-6-medium", "provider": "azure", "model": "claude-sonnet-4-6",    "effort": "medium"},
    {"label": "claude-opus-4-6-low",      "provider": "azure", "model": "claude-opus-4-6",      "effort": "low"},
    {"label": "claude-opus-4-6-medium",   "provider": "azure", "model": "claude-opus-4-6",      "effort": "medium"},
    {"label": "claude-opus-4-7-low",      "provider": "azure", "model": "claude-opus-4-7",      "effort": "low"},
    {"label": "claude-opus-4-7-medium",   "provider": "azure", "model": "claude-opus-4-7",      "effort": "medium"},
    {"label": "gpt-5-low",                "provider": "azure", "model": "gpt-5",                "effort": "low"},
    {"label": "gpt-5-medium",             "provider": "azure", "model": "gpt-5",                "effort": "medium"},
]

# ── lc0 multi-server management ───────────────────────────────────────────────
_LCO_SCRIPT = _REPO_ROOT / "lc0_server/scripts/start_lc0_server.sh"


def _lc0_health(port: int, timeout: float = 2.0) -> bool:
    import httpx
    try:
        r = httpx.get(f"http://localhost:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_lc0_servers(n_workers: int, base_port: int = 7100) -> tuple[list[str], list[subprocess.Popen]]:
    """Ensure n_workers lc0 instances are running on consecutive ports.

    Returns (list_of_urls, list_of_procs_we_started).
    For ports that already have a live server we reuse them without starting a
    new process.  Callers are responsible for terminating the returned procs.
    """
    urls: list[str] = []
    started: list[subprocess.Popen] = []
    n = min(n_workers, 8)  # at most 8 GPUs

    for i in range(n):
        port = base_port + i
        urls.append(f"http://localhost:{port}")
        if _lc0_health(port):
            print(f"  [lc0] port {port} (GPU {i}) already running — reusing")
            continue
        # Start a new server on GPU i
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "LC0_PORT": str(port)}
        proc = subprocess.Popen(
            [str(_LCO_SCRIPT)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        print(f"  [lc0] started on GPU {i} port {port} (PID {proc.pid})")

    # Wait for all to be healthy
    if started:
        print(f"  [lc0] waiting for {len(started)} new server(s)…", end="", flush=True)
        deadline = time.time() + 60
        for port in [base_port + i for i in range(n)]:
            while time.time() < deadline:
                if _lc0_health(port):
                    break
                time.sleep(1)
            else:
                print(f"\n  [lc0] WARNING: port {port} never became healthy")
        print(" ready")

    return urls, started


# ── Answer parser ─────────────────────────────────────────────────────────────
_ANSWER_RE = re.compile(
    r"the popularity is\s+(-?\d+)\s+and the elo is\s+(\d+)",
    re.IGNORECASE,
)


def _parse_answer(text: str) -> tuple[int | None, int | None]:
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None, None
    m = matches[-1]
    return int(m.group(1)), int(m.group(2))


# ── Per-sample evaluation ─────────────────────────────────────────────────────

def _eval_sample(
    idx: int,
    row: dict,
    cfg: dict,
    lc0_url: str,
) -> dict:
    fen: str = row["fen"]
    prompt: list[dict] = row["prompt"]
    ground_truth: dict = row["reward_model"]["ground_truth"]
    true_pop: int = ground_truth["popularity"]
    true_elo: int = ground_truth["elo"]

    system_prompt: str = prompt[0]["content"]
    user_query: str = prompt[1]["content"]

    result: dict = {
        "idx": idx,
        "label": cfg["label"],
        "provider": cfg["provider"],
        "model": cfg["model"],
        "effort": cfg.get("effort"),
        "fen": fen,
        "pred_pop": None,
        "pred_elo": None,
        "true_pop": true_pop,
        "true_elo": true_elo,
        "pop_err": None,
        "elo_err": None,
        "parse_ok": False,
        "elapsed_s": None,
        "tool_calls": None,
        "llm_rounds": None,
        "trace": None,
        "error": None,
    }

    t0 = time.time()
    try:
        analyst = ChessAnalyst(
            model=cfg["model"],
            lc0_url=lc0_url,
            system_prompt=system_prompt,
            provider=cfg["provider"],
            reasoning_effort=cfg.get("effort"),
        )
        answer = analyst.run(user_query, verbose=False)
        elapsed = time.time() - t0

        stats = analyst.last_stats
        pred_pop, pred_elo = _parse_answer(answer)
        parse_ok = pred_pop is not None and pred_elo is not None

        result.update(
            pred_pop=pred_pop,
            pred_elo=pred_elo,
            pop_err=abs(pred_pop - true_pop) if parse_ok else None,
            elo_err=abs(pred_elo - true_elo) if parse_ok else None,
            elapsed_s=round(elapsed, 2),
            tool_calls=stats.get("tool_calls"),
            llm_rounds=stats.get("llm_rounds"),
            parse_ok=parse_ok,
            trace=analyst.last_trace,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        result.update(elapsed_s=round(elapsed, 2), error=str(exc))

    return result


# ── Per-config runner ─────────────────────────────────────────────────────────

def _run_config(
    cfg: dict,
    rows: list[dict],
    lc0_urls: list[str],
    workers: int,
    out_dir: Path,
) -> tuple[str, dict, list[dict]]:
    label = cfg["label"]
    print(f"\n{'='*64}")
    print(f"  {label}  ({len(rows)} samples, {workers} workers, {len(lc0_urls)} lc0 server(s))")
    print(f"{'='*64}")

    results: list[dict] = [None] * len(rows)  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_eval_sample, idx, row, cfg, lc0_urls[idx % len(lc0_urls)]): idx
            for idx, row in enumerate(rows)
        }
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {
                    "idx": idx, "label": label,
                    "provider": cfg["provider"], "model": cfg["model"],
                    "effort": cfg.get("effort"), "fen": rows[idx].get("fen", ""),
                    "pred_pop": None, "pred_elo": None,
                    "true_pop": None, "true_elo": None,
                    "pop_err": None, "elo_err": None,
                    "parse_ok": False, "elapsed_s": None,
                    "tool_calls": None, "llm_rounds": None,
                    "trace": None, "error": str(exc),
                }
            results[idx] = res
            completed += 1
            status = "ok" if res["parse_ok"] else ("err" if res["error"] else "no-parse")
            print(
                f"  [{completed:>3}/{len(rows)}] idx={idx:>3}  {status}"
                + (f"  pop={res['pred_pop']}  elo={res['pred_elo']}" if res["parse_ok"] else "")
                + (f"  ERROR: {str(res['error'])[:60]}" if res["error"] else ""),
                flush=True,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{label}_{timestamp}.jsonl"
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"\n  Saved {len(results)} records → {out_path}")

    metrics = compute_metrics(results)
    print_metrics(label, metrics)
    return label, metrics, results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw-output eval across models and reasoning efforts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=100, metavar="N",
                   help="Number of val samples (first N rows)")
    p.add_argument("--workers", type=int, default=2,
                   help="Concurrent workers per config entry; equals lc0 servers started (1 per GPU)")
    p.add_argument("--lc0-base-port", type=int, default=7100,
                   help="Base port for lc0 servers; workers use ports base..base+workers-1")
    p.add_argument("--out-dir", default="results/raw")
    p.add_argument("--labels", nargs="+", metavar="LABEL",
                   help="Run only these labels from EVAL_CONFIG (default: all)")
    p.add_argument("--parallel-configs", action="store_true",
                   help="Run all configs concurrently (default: sequential)")
    p.add_argument("--split", choices=["train", "test"], default="test",
                   help="Processed HF split to evaluate")
    p.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED,
                   help="Seed for selecting the evaluated samples")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ds = load_processed_split(args.split)
    n = min(args.n, len(ds))
    sampled_rows = sample_rows(list(ds), n_samples=n, seed=args.seed)

    rows: list[dict] = []
    for row in sampled_rows:
        prompt = row["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        rows.append({
            "fen": row["fen"],
            "prompt": prompt,
            "reward_model": row["reward_model"],
            "data_source": row.get("data_source", ""),
        })

    active_configs = EVAL_CONFIG
    if args.labels:
        label_set = set(args.labels)
        active_configs = [c for c in EVAL_CONFIG if c["label"] in label_set]
        if not active_configs:
            print(f"[ERROR] No configs matched labels: {args.labels}")
            print(f"Available: {[c['label'] for c in EVAL_CONFIG]}")
            sys.exit(1)

    out_dir = Path(args.out_dir)

    # Start lc0 servers (one per worker, one per GPU)
    print(f"Loaded {n} samples from ssingh22/llamia-verl-data/puzzle_popularity_elo:{args.split}")
    print(f"Configs:  {[c['label'] for c in active_configs]}")
    print(f"Workers:  {args.workers} per config (1 lc0 server per GPU)")
    print(f"Out dir:  {out_dir}")
    lc0_urls, _started_procs = _ensure_lc0_servers(args.workers, base_port=args.lc0_base_port)
    print(f"lc0 URLs: {lc0_urls}")

    all_results: dict[str, tuple[dict, list[dict]]] = {}

    try:
        if args.parallel_configs and len(active_configs) > 1:
            with ThreadPoolExecutor(max_workers=len(active_configs)) as pool:
                futures = {
                    pool.submit(_run_config, cfg, rows, lc0_urls, args.workers, out_dir): cfg["label"]
                    for cfg in active_configs
                }
                for fut in as_completed(futures):
                    lbl = futures[fut]
                    try:
                        label, metrics, results = fut.result()
                        all_results[label] = (metrics, results)
                    except Exception as exc:
                        print(f"[ERROR] {lbl}: {exc}")
        else:
            for cfg in active_configs:
                label, metrics, results = _run_config(cfg, rows, lc0_urls, args.workers, out_dir)
                all_results[label] = (metrics, results)
    finally:
        for proc in _started_procs:
            proc.terminate()

    if len(all_results) > 1:
        comparison_table(all_results)


if __name__ == "__main__":
    main()
