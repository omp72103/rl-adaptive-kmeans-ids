# AGENTS.md — Base44 dev environment notes

## What this app is

A Python FastAPI backend (`api/app.py`) that runs a real scikit-learn / scipy
RL-adaptive K-Means pipeline against the NSL-KDD intrusion-detection dataset.
The backend serves both the API (`/api/...`) and a single-file static HTML
dashboard (`/app/`, `frontend/index.html` with inline Plotly). No database, no
external services, no secrets required — all data files ship in `data/`.

## Why the original commit failed to start in Base44

The repo had no `docker-compose.base44.yml` or `.base44/environment.json`, so
the Base44 platform had no way to boot it. The commit itself is a complete,
working app — the failure was purely a missing dev-environment setup.

## Fixes applied for the preview

1. **Frontend API URL**: `frontend/index.html` had `apiBase` defaulting to
   `http://localhost:8000`, which doesn't resolve from a remote browser. Changed
   the default to empty string so API calls use relative URLs (same origin as
   the FastAPI server that serves the frontend).
2. **Root redirect**: Added `GET /` → `RedirectResponse('/app/')` in
   `api/app.py` so the preview's initial load lands on the dashboard instead
   of a 404.

## How to run

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

- Web entry point: host port 3000 → container port 8000 (uvicorn --reload)
- Health check: `GET /api/health` → `{"status":"ok"}`
- Dashboard: `http://localhost:3000/app/` (or `/` redirects there)

## Verifying it works

- `curl http://localhost:3000/api/health` → `{"status":"ok"}`
- `curl http://localhost:3000/api/dataset/info` → real row counts from NSL-KDD
- Open the dashboard, click "Run real clustering" — returns real metrics
