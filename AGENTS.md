# Base44 Dev Environment

## What this app is
A Python FastAPI backend (`api/app.py`) running an RL-adaptive K-Means
intrusion-detection pipeline against the real NSL-KDD dataset. It serves
both the API (`/api/*`) and a static dashboard (`/app/`) from a single
uvicorn process.

## How to run it
```bash
docker compose -f docker-compose.base44.yml up -d
```
- Web entry point: port 3000 → container port 8000
- Dashboard URL: `/app/`
- Health check: `/api/health`
- No external secrets required — all data is local (`data/KDDTrain+.txt`, `data/KDDTest+.txt`)

## Key details
- The frontend (`frontend/index.html`, ~4.8MB single file with embedded
  Plotly/d3) has an "API base URL" input that defaults to `""` (empty)
  so fetch calls use relative URLs — the API and frontend are served
  from the same origin.
- Dependencies install on container startup via `pip install -r requirements.txt`
  (numpy, pandas, scikit-learn, scipy, fastapi, uvicorn, etc.).
- The first boot takes ~30-40s for pip install before uvicorn starts.
- `src/pipeline.py` runs real scikit-learn/scipy code against the real
  dataset on every request — nothing is precomputed or mocked.
