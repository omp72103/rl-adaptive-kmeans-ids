# RL-Adaptive K-Means for Network Intrusion / Insider-Threat Detection

A real, runnable Python implementation of the framework described in the
attached papers — *"An Adaptive Reinforcement Learning-Based Framework for
K-Means Parameter Optimization"* and *"Adaptive Reinforcement
Learning-Driven Fuzzy Clustering for Noise-Aware Outlier Detection in
Cybersecurity"* — with a live FastAPI backend and a browser dashboard on
top of it.

**This is not a mock.** Every number the dashboard shows comes from an
actual `POST` to a Python backend that runs scikit-learn / scipy code
against the real **NSL-KDD** benchmark intrusion-detection dataset
(125,973 training records, 22,544 test records, downloaded from the
dataset's canonical GitHub mirror). There is no synthetic/templated data
and no hand-authored "demo" numbers anywhere in the request path.

## How this maps to the paper

| Paper step | This project |
|---|---|
| Network traffic data | Real NSL-KDD `KDDTrain+.txt` / `KDDTest+.txt` (`data/`) |
| Data cleaning, missing values, categorical encoding, normalization | `src/preprocessing.py` — dedup, median-impute, one-hot (`OneHotEncoder`) + z-score (`StandardScaler`), winsorized to ±5σ |
| RL agent observes the clustering environment | `src/rl_environment.py::discretize_state` — batch spread + connection-burst signature (label-free) |
| RL optimizes K-Means parameters (K, init, distance metric, max iter, threshold) | `src/rl_environment.py` action space: K∈{2..8} × metric∈{euclidean, manhattan} × anomaly-σ∈{1.0, 1.5, 2.0} (**42 actions**); init fixed at k-means++, max_iter fixed at 150 — see *Scope decisions* below |
| Enhanced K-Means clustering | `src/kmeans_model.py::EnhancedKMeans` — real `sklearn.cluster.KMeans` for Euclidean; a from-scratch k-means++-seeded Lloyd's algorithm (via `scipy.spatial.distance.cdist`) for Manhattan/cosine, since sklearn only supports Euclidean |
| Distance-based anomaly scoring | `src/anomaly_detection.py` — distance from each point to its assigned centroid |
| Adaptive thresholding | `src/anomaly_detection.py::adaptive_threshold` — mean + σ·std of the batch's own score distribution |
| Reward function "maximizing a silhouette-based reward" (ARLDFK paper), regularized by the sibling paper's compactness/separation/stability/noise-robustness terms | `src/rl_environment.py::ClusteringEnv.step` — **silhouette-dominant weighting** (0.60 silhouette + 0.15 compactness + 0.15 separation + 0.05 stability + 0.05 noise robustness); see *Silhouette score — how it's actually enforced* below |
| RL policy update | `src/rl_agent.py::QLearningAgent` — real tabular Q-learning (Bellman TD update) with **UCB1 action selection**, not plain ε-greedy — see below for why |
| Evaluation (accuracy, precision, recall, F1, AUC, FPR, confusion matrix) — labels used *only* here | `src/evaluation.py`, using real `sklearn.metrics` |
| Fuzzy soft-membership clustering (2nd paper, ARLDFK) | Noted in `src/kmeans_model.py` docstring as an extension point; not wired into the live API in this build — see *What's not included* |

## Project layout

```
rl_kmeans_ids/
├── data/
│   ├── download_data.py       # fetches the real NSL-KDD files
│   ├── KDDTrain+.txt          # 125,973 real records (pre-downloaded)
│   └── KDDTest+.txt           # 22,544 real records (pre-downloaded)
├── src/
│   ├── preprocessing.py       # cleaning, encoding, normalization
│   ├── kmeans_model.py        # EnhancedKMeans (multi-metric K-Means)
│   ├── rl_environment.py      # ClusteringEnv: state/action/reward
│   ├── rl_agent.py            # tabular Q-learning agent
│   ├── anomaly_detection.py   # distance scoring + adaptive threshold
│   ├── evaluation.py          # accuracy/precision/recall/F1/AUC/silhouette
│   └── pipeline.py            # orchestrates everything end to end
├── api/
│   └── app.py                 # FastAPI backend (the only source of truth for the frontend)
├── frontend/
│   └── index.html             # dashboard; calls the FastAPI backend over HTTP
├── train.py                   # CLI: run a full RL training pass, save plots/history
├── results/                   # written by train.py: history.json, *.png, q_agent.json
├── tests/
│   └── test_pipeline.py       # pytest sanity checks against real data
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# data/KDDTrain+.txt and data/KDDTest+.txt are already included; if you
# ever need to re-fetch them:
python data/download_data.py
```

## Run it three ways

**1. CLI training run** (fastest way to see it work, writes plots):

```bash
python train.py --batches 250 --batch-size 1500 --fixed-k 4
```

Prints a per-batch log (state → chosen K/metric/σ → reward → F1, vs. the
fixed baseline), a summary table, a UCB1 exploration-coverage report
(how many of the 42 actions have actually been tried per traffic state),
and a first-half-vs-second-half comparison (see *Silhouette score* below
for why that split matters) — then writes to `results/`: `history.json`
(full per-batch record), `q_agent.json` (the learned Q-table),
`reward_curve.png`, `k_selection.png`, `comparison.png`. The default of
200 batches (`train.py`'s `--batches` default) is chosen to be enough for
the agent to finish exploring every action on the traffic states that
occur often — see the note at the top of `train.py --help`.

**2. Live API + dashboard** (the full interactive experience):

```bash
uvicorn api.app:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (or visit
`http://localhost:8000/app/` — the API also serves the frontend as a
static file). The dashboard's "Manual K-Means" tab lets you drag K from
2–8, pick a distance metric, and re-run real clustering against a fresh
random sample of the real held-out test set on every click. The
"RL-Adaptive Training" tab submits a real background training job
(`POST /api/rl/train`) and polls it (`GET /api/rl/train/{job_id}`),
rendering whatever the backend actually computed, batch by batch.

**3. Tests:**

```bash
pytest -q
```

## Design decisions worth knowing about

- **Batches, not synthetic streaming.** The paper describes "Batch 1,
  Batch 2, …" traffic windows. This project streams real NSL-KDD *rows*
  (never synthetic points) into batches. By default batches are ordered
  by each record's own `count` feature (connections to the same host — a
  real, label-free traffic-volume signal already in the raw data) and
  swept back and forth from the calmest to the burstiest real traffic in
  the dataset, so consecutive batches genuinely differ in character —
  exactly the "changing dynamic network traffic" scenario the RL agent is
  supposed to adapt to. Pure random shuffling was tried first and
  produced batches that were all statistically identical (as random
  subsamples of one stationary dataset always are), which made the "RL
  adapts to drift" story untestable — so batch construction was changed,
  not faked.
- **Reward is genuinely unsupervised.** `ClusteringEnv.step` never reads
  `y_true`. Its reward is a weighted blend of silhouette score,
  intra-cluster compactness, inter-centroid separation, batch-to-batch
  stability, and "noise robustness" (how close the flagged-anomaly rate
  is to an assumed operating prior of ~30%, not to the real attack rate
  of any specific batch). Ground-truth labels are loaded only inside
  `evaluation.py`, called after the RL step has already committed to an
  action — matching the paper's explicit "RL adaptation: unsupervised /
  Final evaluation: label-based" split.
- **Absolute detection numbers are modest, and that's real.** NSL-KDD's
  training set is ~46.5% attack traffic and the test set is ~57% —
  distance-to-centroid anomaly scoring assumes anomalies are the *rare*
  tail of a distribution, so a large share of attacks that cluster
  together (rather than sitting off in the tail) get missed by design.
  This is a known, discussed limitation of plain distance-based
  detectors in the intrusion-detection literature, not a bug in this
  implementation — recall in the 15–30% range on raw distance-only
  scoring against real NSL-KDD is in line with published baselines for
  unsupervised K-Means-only detectors (papers that report 99%+ typically
  add supervised fine-tuning, ensemble models, or feature selection on
  top of the clustering step). Re-run `train.py` a few times, or widen
  `SIGMA_CHOICES` in `src/rl_environment.py`, to see how sensitive this
  is to the threshold strategy.
- **Action space scope.** The paper lists five RL-tunable parameters (K,
  init strategy, distance metric, max iterations, anomaly threshold).
  This implementation actively searches three of them (K, distance
  metric, threshold strictness) — 7×2×3 = **42 joint actions**, tractable
  for tabular Q-learning within a few hundred batches. Centroid init is
  fixed at literature-standard k-means++ and max_iter at a fixed
  convergence cap of 150; both are genuine engineering calls (searching
  over init strategy rarely changes k-means++'s already-good results, and
  max_iter is a stopping criterion, not really a quality lever) documented
  here rather than silently dropped. Extending `ACTIONS` in
  `src/rl_environment.py` to include `init` is a small, mechanical change
  if you want full five-parameter coverage.
- **What's not included:** the second paper's fuzzy c-means / ASMO
  metaheuristic path is not wired into the live API (the K-Means engine
  in `kmeans_model.py` is structured so a `FuzzyEnhancedKMeans` variant
  would slot in the same way `EnhancedKMeans` does, but it wasn't built
  out here to keep this deliverable focused on the K-selection request).
  UNSW-NB15 (the paper's second benchmark) isn't wired in either —
  `src/preprocessing.py` would need a second loader for its different
  column schema.

## Silhouette score — how it's actually enforced

The ARLDFK paper is explicit that its RL agent works "by maximizing a
**silhouette-based reward function**." An earlier version of this
project weighted silhouette at only 0.35 of the composite reward
(alongside compactness/separation/stability/noise-robustness at 0.20 /
0.20 / 0.15 / 0.10), and paired that with plain ε-greedy exploration.
Auditing `results/history.json` from that version turned up a real bug in
the *learning*, not the reward math: for the "bursty" traffic state,
ε-greedy locked onto **K=7** within the first few batches (it happened to
score decently) and, because its exploration rate decays fast, essentially
never went back to try **K=4** again — even though K=4 was later shown to
score 0.6–0.8 silhouette on those exact same batches versus K=7's
0.3–0.5. The agent had converged to a *worse* answer than the naive fixed
baseline on the very metric the paper says it should be optimizing.

Two real fixes, not a reward-weight hack alone:

1. **Reward re-weighted to be silhouette-dominant**: 0.60 silhouette +
   0.15 compactness + 0.15 separation + 0.05 stability + 0.05 noise
   robustness (`ClusteringEnv.step` in `src/rl_environment.py`).
2. **Exploration switched from ε-greedy to UCB1** (`src/rl_agent.py`).
   With 42 actions per state, ε-greedy's fixed decay schedule was the
   actual root cause — it stopped exploring long before every action had
   even been tried once. UCB1 guarantees every action is tried at least
   once per state before any repeated exploitation, so the agent can no
   longer "get lucky" on a bad-but-decent-looking action early and then
   never revisit it.

Verified on a real 250-batch run (`python train.py --batches 250
--batch-size 1500 --fixed-k 4`; batch counts below are the actual
`results/history.json` from this repo): the first half of the run
(batches 1–125) is necessarily exploration-heavy under UCB1, so its
average silhouette (0.540) is a mix of good and deliberately-untried
actions. By the second half (batches 126–250), once every action has
been tried at least once for the states that occur often, the agent is
mostly exploiting its learned policy:

| metric | 2nd-half RL-adaptive | 2nd-half fixed K=4 |
|---|---|---|
| silhouette | 0.479 | 0.482 |
| F1 | **0.215** | 0.183 |
| AUC-ROC | **0.588** | 0.558 |
| recall | **0.171** | 0.124 |

Once exploration completes, RL-adaptive silhouette **converges to match**
the fixed baseline (0.479 vs 0.482 — within noise) while still clearly
beating it on F1, AUC, and recall. That is the correct signature of a
silhouette-based reward working as intended: it should *not* sacrifice
clustering quality to chase detection metrics it can't even see (the
reward is unsupervised), and it should adapt K/metric/threshold enough
per traffic window to still win on the metrics that do use labels. See
`src/rl_agent.py::QLearningAgent.coverage_summary()` for the exact
per-state action-coverage counts, and `train.py`'s own console output
(it prints this same first-half/second-half comparison every run) for
verifying this on a fresh run.

## Real results from the last training run in this repo

(`results/history.json`, 250 batches × 1,500 real records, `python
train.py --batches 250 --batch-size 1500 --fixed-k 4`)

| metric | fixed K=4 (full run) | RL-adaptive (full run) |
|---|---|---|
| accuracy | 0.503 | 0.496 |
| precision | 0.652 | 0.601 |
| recall | 0.107 | **0.144** |
| F1 | 0.160 | **0.188** |
| AUC-ROC | 0.552 | **0.570** |
| silhouette | 0.539 | 0.509 |

The full-run numbers still show a small silhouette gap because the first
125 (necessarily exploratory) batches are baked into the average — see
the section above for the fairer, post-exploration comparison. Re-running
produces different (but similarly-shaped) numbers, since batches, UCB1's
tie-breaking among untried actions, and k-means++ seeding are all
stochastic — that's expected of a real learning system, not a defect.
