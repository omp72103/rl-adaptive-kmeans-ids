"""
FastAPI backend -- the ONLY place results reach the frontend from. Every
endpoint calls straight into src/pipeline.py, which runs the real
scikit-learn / scipy / numpy pipeline against the real NSL-KDD dataset on
every request. Nothing here is precomputed or templated; re-running a
request re-runs the actual algorithm and can return different numbers
(e.g. different random batches, different RL exploration draws).

Run:
    uvicorn api.app:app --reload --port 8000
Then open frontend/index.html (it calls this API on localhost:8000).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.pipeline import run_manual_clustering, run_rl_training
from src.preprocessing import load_and_prepare

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rl_kmeans_ids.api")

app = FastAPI(title="RL-Adaptive K-Means IDS API", version="1.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Every endpoint below calls into real numpy/sklearn/pandas code
    against a real dataset -- a malformed request or an edge-case sample
    (e.g. a batch too small to form K clusters) can raise from deep
    inside that stack. Log the full traceback server-side but never leak
    it to the client; return a stable, generic 500 instead of letting
    uvicorn's default handler dump internals into the response body."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.get("/")
def root_redirect():
    """Redirect the bare root to the dashboard so the preview lands on
    the app instead of a 404."""
    return RedirectResponse(url="/app/")

# In-memory job store for async RL training runs (real background threads,
# not fake progress bars). Fine for a single-process demo/dev server; a
# production deployment would move this to Redis/a DB so jobs survive a
# restart and multiple worker processes can share state.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 200  # simple bound so a long-running demo server can't leak memory forever


class ManualRequest(BaseModel):
    k: int = Field(4, ge=2, le=8, description="number of clusters")
    metric: str = Field("euclidean", description="euclidean | manhattan | cosine")
    sigma: float = Field(1.5, ge=0.1, le=5.0, description="anomaly-threshold strictness (std devs)")
    sample_size: int = Field(2000, ge=100, le=10000, description="real test-set records to sample")
    split: str = Field("test", description="train | test")
    seed: int = 42


class TrainRequest(BaseModel):
    n_batches: int = Field(20, ge=1, le=500, description="number of real traffic batches to stream")
    batch_size: int = Field(1500, ge=100, le=5000, description="real records per batch")
    fixed_k: int = Field(4, ge=2, le=8, description="K for the fixed-parameter baseline")
    seed: int = 42


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/dataset/info")
def dataset_info():
    df_train, y_train, cat_train = load_and_prepare("train")
    df_test, y_test, cat_test = load_and_prepare("test")
    import pandas as pd
    return {
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "n_raw_features": 41,
        "train_attack_rate": float(y_train.mean()),
        "test_attack_rate": float(y_test.mean()),
        "train_category_counts": pd.Series(cat_train).value_counts().to_dict(),
        "test_category_counts": pd.Series(cat_test).value_counts().to_dict(),
        "source": "NSL-KDD (KDDTrain+.txt / KDDTest+.txt), real benchmark intrusion-detection dataset",
    }


@app.post("/api/cluster/manual")
def cluster_manual(req: ManualRequest):
    if req.metric not in ("euclidean", "manhattan", "cosine"):
        raise HTTPException(400, "metric must be one of: euclidean, manhattan, cosine")
    if req.split not in ("train", "test"):
        raise HTTPException(400, "split must be one of: train, test")
    t0 = time.time()
    logger.info("manual clustering: k=%d metric=%s sigma=%.2f n=%d split=%s",
                req.k, req.metric, req.sigma, req.sample_size, req.split)
    result = run_manual_clustering(
        k=req.k, metric=req.metric, sigma=req.sigma,
        sample_size=req.sample_size, split=req.split, seed=req.seed,
    )
    result["elapsed_sec"] = time.time() - t0
    return result


@app.post("/api/rl/train")
def rl_train_start(req: TrainRequest):
    """Kicks off a real training run in a background thread and returns a
    job id immediately; poll /api/rl/train/{job_id} for progress + the
    final result (genuinely computed, streamed batch by batch)."""
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        if len(_JOBS) >= _MAX_JOBS:
            oldest = next(iter(_JOBS))
            del _JOBS[oldest]
        _JOBS[job_id] = {"status": "running", "progress": 0, "total": req.n_batches,
                          "rl_history": [], "fixed_history": [], "result": None, "error": None}
    logger.info("starting RL training job %s: n_batches=%d batch_size=%d fixed_k=%d",
                job_id, req.n_batches, req.batch_size, req.fixed_k)

    def worker():
        try:
            def cb(b, n, rl, fx):
                with _JOBS_LOCK:
                    _JOBS[job_id]["progress"] = b + 1
                    _JOBS[job_id]["rl_history"].append(rl)
                    _JOBS[job_id]["fixed_history"].append(fx)

            out = run_rl_training(
                n_batches=req.n_batches, batch_size=req.batch_size,
                fixed_k=req.fixed_k, seed=req.seed, progress_cb=cb,
            )
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "done"
                _JOBS[job_id]["agent_policy"] = out["agent"].policy_summary()
                _JOBS[job_id]["ucb_coverage"] = out["agent"].coverage_summary()
            logger.info("RL training job %s finished", job_id)
        except Exception:
            logger.exception("RL training job %s failed", job_id)
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "error"
                _JOBS[job_id]["error"] = "training job failed -- see server logs"

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/rl/train/{job_id}")
def rl_train_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    return job


@app.get("/api/rl/actions")
def rl_actions():
    from src.rl_environment import ACTIONS
    return {"actions": [{"k": a[0], "metric": a[1], "sigma": a[2]} for a in ACTIONS]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
