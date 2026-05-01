"""Evaluation metrics for puzzle ELO and popularity prediction.

Metric families
---------------
1. Accuracy          |pred - true| ≤ threshold  (primary goal metric)
2. MAE               mean absolute error
3. RMSE              root mean squared error
4. R²                coefficient of determination
5. Weighted Kappa    quadratic Cohen's κ over ordinal bins
6. Interval coverage fraction within ±k·RD, where RD = σ of the test distribution
7. Spearman's ρ      rank correlation

RD note
-------
Lichess puzzle ratings carry a Glicko RD (Rating Deviation).  That field is
not included in puzzle_val.parquet, so we use the test-set standard deviation
as a proxy (ELO≈542, Pop≈15.5).  A well-calibrated predictor should achieve
≥68 % coverage at 1 RD, ≥95 % at 2 RD.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats

# ── Reference constants (computed from full puzzle_val.parquet) ───────────────
ELO_RD: float = 542.3   # σ of true ELO — used as RD proxy
POP_RD: float = 15.5    # σ of true popularity

# Primary-metric accuracy thresholds
ACC_ELO: float = 200.0  # ±200 ELO points  (~0.37 σ)
ACC_POP: float = 15.0   # ±15 pop points   (~1 σ)

# Ordinal bins for weighted kappa
# ELO: 5 chess skill bands  (bullet/blitz/rapid/club/master)
ELO_EDGES: list[float] = [1000.0, 1500.0, 2000.0, 2500.0]
# Pop: 5 bands fitted to skewed distribution (mean≈85, most values >75)
POP_EDGES: list[float] = [0.0, 50.0, 75.0, 90.0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def _quadratic_kappa(y_true: np.ndarray, y_pred: np.ndarray, edges: list[float]) -> float:
    """Cohen's quadratic weighted kappa over digitised categories.

    ``edges`` defines the bin boundaries (exclusive on the right), yielding
    ``len(edges) + 1`` ordinal categories via ``np.digitize``.
    Predictions are clipped to the range of ``edges`` before digitising.
    """
    lo, hi = edges[0], edges[-1]
    y_pred_c = np.clip(y_pred, lo - 1e-9, hi + 1e-9)

    cats_true = np.digitize(y_true, edges)
    cats_pred = np.digitize(y_pred_c, edges)
    n_cats = len(edges) + 1
    n = len(y_true)

    # Confusion matrix
    cm = np.zeros((n_cats, n_cats), dtype=float)
    for t, p in zip(cats_true, cats_pred):
        cm[t, p] += 1.0

    # Quadratic weight matrix: w[i,j] = ((i-j)/(n_cats-1))^2
    idx = np.arange(n_cats)
    w = ((idx[:, None] - idx[None, :]) / max(n_cats - 1, 1)) ** 2

    row_sum = cm.sum(axis=1, keepdims=True)
    col_sum = cm.sum(axis=0, keepdims=True)
    expected = (row_sum @ col_sum) / n

    num = float((w * cm).sum())
    den = float((w * expected).sum())
    if den == 0:
        return float("nan")
    return 1.0 - num / den


def _safe(fn, *args) -> float:
    try:
        v = fn(*args)
        return float("nan") if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), 4)
    except Exception:
        return float("nan")


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute all metric families from a list of eval result dicts.

    Each dict must contain: ``parse_ok``, ``pred_pop``, ``pred_elo``,
    ``true_pop``, ``true_elo``, ``elapsed_s``, ``tool_calls``.
    Optional: ``trace`` (list of turn dicts with ``thinking`` key).
    """
    n = len(results)
    parsed = [r for r in results if r.get("parse_ok")]
    np_ = len(parsed)
    nan = float("nan")

    elapsed = [r["elapsed_s"] for r in results if r.get("elapsed_s") is not None]
    tool_list = [r["tool_calls"] for r in results if r.get("tool_calls") is not None]
    thinking_chars = [
        sum(len(t.get("thinking", "")) for t in r["trace"])
        for r in results if r.get("trace")
    ]

    base = {
        "n": n,
        "parse_ok": np_,
        # placeholders — overwritten below if we have parsed results
        "pop_acc": nan, "elo_acc": nan,
        "pop_MAE": nan, "elo_MAE": nan,
        "pop_RMSE": nan, "elo_RMSE": nan,
        "pop_R2": nan, "elo_R2": nan,
        "pop_kappa": nan, "elo_kappa": nan,
        "pop_cov1RD": nan, "pop_cov2RD": nan,
        "elo_cov1RD": nan, "elo_cov2RD": nan,
        "pop_spearman": nan, "elo_spearman": nan,
        "avg_tools": round(sum(tool_list) / len(tool_list), 2) if tool_list else nan,
        "avg_s": round(sum(elapsed) / len(elapsed), 2) if elapsed else nan,
        "avg_thinking_chars": round(sum(thinking_chars) / len(thinking_chars), 0) if thinking_chars else nan,
    }

    if not parsed:
        return base

    pp = np.array([r["pred_pop"] for r in parsed], dtype=float)
    tp = np.array([r["true_pop"] for r in parsed], dtype=float)
    pe = np.array([r["pred_elo"] for r in parsed], dtype=float)
    te = np.array([r["true_elo"] for r in parsed], dtype=float)

    pop_err = np.abs(pp - tp)
    elo_err = np.abs(pe - te)

    base.update({
        # 1. Accuracy
        "pop_acc":  round(float(np.mean(pop_err <= ACC_POP)), 4),
        "elo_acc":  round(float(np.mean(elo_err <= ACC_ELO)), 4),
        # 2. MAE
        "pop_MAE":  round(float(pop_err.mean()), 2),
        "elo_MAE":  round(float(elo_err.mean()), 2),
        # 3. RMSE
        "pop_RMSE": round(float(np.sqrt((pop_err ** 2).mean())), 2),
        "elo_RMSE": round(float(np.sqrt((elo_err ** 2).mean())), 2),
        # 4. R²
        "pop_R2":   _safe(_r2, tp, pp),
        "elo_R2":   _safe(_r2, te, pe),
        # 5. Weighted Kappa
        "pop_kappa": _safe(_quadratic_kappa, tp, pp, POP_EDGES),
        "elo_kappa": _safe(_quadratic_kappa, te, pe, ELO_EDGES),
        # 6. Interval coverage vs. RD  (1σ and 2σ of the test distribution)
        "pop_cov1RD": round(float(np.mean(pop_err <= POP_RD)), 4),
        "pop_cov2RD": round(float(np.mean(pop_err <= 2 * POP_RD)), 4),
        "elo_cov1RD": round(float(np.mean(elo_err <= ELO_RD)), 4),
        "elo_cov2RD": round(float(np.mean(elo_err <= 2 * ELO_RD)), 4),
        # 7. Spearman's ρ
        "pop_spearman": _safe(lambda a, b: stats.spearmanr(a, b)[0], tp, pp),
        "elo_spearman": _safe(lambda a, b: stats.spearmanr(a, b)[0], te, pe),
    })
    return base


def print_metrics(label: str, m: dict) -> None:
    """Print a human-readable metrics block."""
    print(f"  ── {label} ──")
    print(
        f"    parse {m['parse_ok']}/{m['n']}"
        f"  pop_acc={m['pop_acc']}  elo_acc={m['elo_acc']}"
    )
    print(
        f"    MAE   pop={m['pop_MAE']}  elo={m['elo_MAE']}"
        f"   RMSE  pop={m['pop_RMSE']}  elo={m['elo_RMSE']}"
    )
    print(
        f"    R²    pop={m['pop_R2']}  elo={m['elo_R2']}"
        f"   κ     pop={m['pop_kappa']}  elo={m['elo_kappa']}"
    )
    print(
        f"    cov@1RD  pop={m['pop_cov1RD']}  elo={m['elo_cov1RD']}"
        f"   cov@2RD  pop={m['pop_cov2RD']}  elo={m['elo_cov2RD']}"
    )
    print(
        f"    ρ     pop={m['pop_spearman']}  elo={m['elo_spearman']}"
        f"   tools={m['avg_tools']}  s={m['avg_s']}"
        f"  think={m['avg_thinking_chars']}"
    )


def comparison_table(all_results: dict[str, tuple[dict, list]]) -> None:
    """Print a compact cross-model comparison table."""
    cols = [
        ("Label",       "<40"),
        ("ok",          ">4"),
        ("elo_acc",     ">8"),
        ("elo_MAE",     ">8"),
        ("elo_RMSE",    ">9"),
        ("elo_R2",      ">7"),
        ("elo_kappa",   ">9"),
        ("elo_cov1RD",  ">10"),
        ("elo_spear",   ">9"),
        ("pop_acc",     ">8"),
        ("pop_MAE",     ">8"),
        ("pop_spear",   ">9"),
        ("avg_s",       ">7"),
        ("think",       ">8"),
    ]
    header = "  ".join(f"{name:{fmt}}" for name, fmt in cols)
    print(f"\n{'='*len(header)}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for lbl, (m, _) in all_results.items():
        row = [
            (lbl,                     "<40"),
            (m["parse_ok"],           ">4"),
            (m["elo_acc"],            ">8"),
            (m["elo_MAE"],            ">8"),
            (m["elo_RMSE"],           ">9"),
            (m["elo_R2"],             ">7"),
            (m["elo_kappa"],          ">9"),
            (m["elo_cov1RD"],         ">10"),
            (m["elo_spearman"],       ">9"),
            (m["pop_acc"],            ">8"),
            (m["pop_MAE"],            ">8"),
            (m["pop_spearman"],       ">9"),
            (m["avg_s"],              ">7"),
            (m["avg_thinking_chars"], ">8"),
        ]
        print("  ".join(f"{str(v):{fmt}}" for v, fmt in row))
