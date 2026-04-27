"""HTTP frontend for an lc0 inference server.

Run via scripts/start_lc0_server.sh, which sets:
    LC0_BIN          path to lc0 binary
    LC0_NET          path to .pb.gz weights
    LC0_BACKEND      cuda-fp16 / cuda / cudnn-fp16 / blas / ...
    LC0_THREADS      lc0 search threads
    LC0_HOST/PORT    bind address

The server holds a single Lc0Engine. lc0 itself is internally multithreaded;
HTTP-level concurrency is serialized by Lc0Engine's lock.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import Lc0Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lc0_server")

ENGINE: Lc0Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    binary = os.environ["LC0_BIN"]
    weights = os.environ["LC0_NET"]
    backend = os.environ.get("LC0_BACKEND", "cuda-fp16")
    threads = int(os.environ.get("LC0_THREADS", "2"))
    minibatch = int(os.environ.get("LC0_MINIBATCH", "128"))
    max_prefetch = int(os.environ.get("LC0_MAX_PREFETCH", "32"))
    nncache = int(os.environ.get("LC0_NNCACHE", "200000"))
    ENGINE = Lc0Engine(
        binary=binary,
        weights=weights,
        backend=backend,
        threads=threads,
        extra_options={
            "MinibatchSize": minibatch,
            "MaxPrefetch": max_prefetch,
            "NNCacheSize": nncache,
        },
    )
    logger.info("lc0 ready: id=%s backend=%s", dict(ENGINE.id), backend)
    try:
        yield
    finally:
        if ENGINE is not None:
            ENGINE.close()


app = FastAPI(title="lc0-server", version="0.1.0", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    fen: str = Field(..., description="FEN of the position (or starting FEN if `moves` is given)")
    moves: list[str] | None = Field(
        None,
        description="Optional UCI move list applied on top of `fen` to fill BT4's history planes faithfully.",
    )
    nodes: int | None = Field(None, ge=1, description="Search node budget")
    movetime_ms: int | None = Field(None, ge=1, description="Search wall-time in ms")
    multipv: int = Field(1, ge=1, le=64, description="Number of principal variations")


class PolicyRequest(BaseModel):
    fen: str
    moves: list[str] | None = Field(
        None,
        description="Optional UCI move list applied on top of `fen` (recommended for raw-policy queries).",
    )
    nodes: int | None = Field(
        None,
        ge=1,
        description="Search budget. Default: max(n_legal_moves+2, 8) so every child gets at least one visit.",
    )
    policy_temperature: float | None = Field(
        None,
        gt=0.0,
        le=10.0,
        description="Override PolicyTemperature for this call. Does not change argmax(P), only the distribution shape.",
    )


@app.get("/health")
def health() -> dict:
    if ENGINE is None:
        raise HTTPException(503, "engine not ready")
    return {
        "status": "ok",
        "engine_id": dict(ENGINE.id),
        "backend": ENGINE.backend,
        "weights": os.path.basename(ENGINE.weights),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if ENGINE is None:
        raise HTTPException(503, "engine not ready")
    return ENGINE.analyse(
        fen=req.fen,
        nodes=req.nodes,
        movetime_ms=req.movetime_ms,
        multipv=req.multipv,
        moves=req.moves,
    )


@app.post("/policy")
def policy(req: PolicyRequest) -> dict:
    if ENGINE is None:
        raise HTTPException(503, "engine not ready")
    return ENGINE.policy(
        fen=req.fen,
        nodes=req.nodes,
        moves=req.moves,
        policy_temperature=req.policy_temperature,
    )
