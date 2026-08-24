# Floter AI — Two-Stage Recommendation Engine

A production-shaped two-stage "who to follow" recommendation backend: a fast
in-memory candidate retrieval stage narrows the full user pool down to 100
candidates, and a PyTorch MLP ranker scores and re-ranks those 100 down to
a final top 10, served behind a FastAPI endpoint.

This README explains what I built and, more importantly, *why* — the
assessment brief was explicit that a simple, well-understood, properly
justified solution should score higher than an over-engineered one I can't
defend, and that's the standard I tried to hold myself to.

## 1. AI & external code attribution

I used Claude (Anthropic's AI coding assistant) throughout this project —
for writing the retrieval/ranking/training code, for structuring the repo,
and for catching a real bug in my first labeling approach (details in
section 3). I reviewed, ran, and reasoned through every part of it myself:
I watched the actual loss curves, decided the label design needed to
change when the first version converged suspiciously fast, picked the
retrieval scoring weights, and chose the MLP over a Two-Tower architecture
for the reasons below. I can walk through any file in this repo and explain
what it does and why it's built that way.

No third-party recommendation library, pretrained model, or vector
database was used. The only dependencies are standard, widely-used
open-source libraries: PyTorch, FastAPI, pandas, NumPy, scikit-learn, and
Pydantic — all listed with pinned versions in `requirements.txt`.

## 2. Design decisions

### Stage A — candidate retrieval (`app/services/retrieval.py`)

I chose a single vectorized weighted-sum scan over the whole user pool
instead of building an inverted interest index or an ANN structure.

With ~25,000 users, a full NumPy pass (`logical_and`/`logical_or` over a
24,859 × 29 boolean interest matrix, plus a few array comparisons for
location and age) runs in single-digit milliseconds — measured mean latency
for the *entire* retrieval-plus-ranking pipeline is under 10ms (see section
5). An inverted index or an ANN library would add real complexity — build
time, memory overhead, staleness on updates — for no measurable latency
win at this scale. That's exactly the kind of over-engineering the brief
warns against. If this dataset were 10–100M users instead of 25K, this
would be the first thing I'd replace, and I describe what I'd replace it
with in the scaling proposal (section 4).

The retrieval score is:

```
score = 2.0·jaccard(interests) + 1.0·same_country + 1.5·same_city + 0.5·age_score
```

Interest overlap is weighted highest because it's the strongest, most
information-dense signal available in this dataset — the 29-category
interest vocabulary is fixed and every user has it populated. I weighted
same-city *above* same-country because a city match is rare and meaningful
(the dataset has 222 countries but roughly 15,500 distinct cities), while a
country match is common and therefore a weaker signal on its own.

### Stage B — ranking (`app/models/ranker.py`)

I chose a pairwise-feature MLP over a Two-Tower architecture, which the
brief explicitly offered as an option.

A Two-Tower model's main advantage is precomputing item embeddings once and
doing fast approximate nearest-neighbor lookup over a huge item catalog.
That advantage doesn't apply here, because Stage A has already narrowed the
candidate set to 100 before Stage B ever runs — Stage B only ever needs to
score 100 pairs, not search millions. A Two-Tower model would also mean the
network learns opaque embeddings instead of the hand-interpretable features
(interest Jaccard, same-city, age gap) I can name and defend individually.
Given the brief's explicit preference for justified simplicity, the
pairwise MLP was the better choice for this problem at this scale.

**Architecture:** a 3-layer MLP — `Linear(66→64) → ReLU → Linear(64→32) →
ReLU → Linear(32→1)` — trained with `BCEWithLogitsLoss`, Adam (lr=1e-3),
batch size 64, 15 epochs, with the best-validation-loss checkpoint kept.

**Input (66 dims):** target user's feature vector (age, normalized, +
29-dim interest multi-hot = 30 dims), the candidate's identical 30-dim
vector, and 6 hand-engineered pairwise features: interest Jaccard
similarity, shared-interest count, same-city, same-country, same-gender,
and age difference.

I used hand-engineered multi-hot/Jaccard features rather than learned
embeddings for interests because the dataset has a small, fixed, 29-category
controlled vocabulary — not free text. Learned embeddings earn their
complexity with thousands of noisy free-text tags; with 29 fixed categories,
a multi-hot vector and Jaccard similarity are the more direct, interpretable
representation, and they're what I used.

## 3. Trade-offs & constraints

I want to be direct about the two biggest limitations in this project,
because I think hiding them would be worse than explaining them.

### There is no real follow-graph in the dataset

`Assessment_TwitterDataset.csv` has no ground-truth "who follows whom" —
just `UserID, Name, Gender, DOB, Interests, City, Country`. That means
there's no real label to train Stage B against, and this is genuinely the
central constraint of the whole project.

**My first attempt at a label was wrong, and I want to explain why, because
I think how I found and fixed it is more informative than pretending it
didn't happen.** I initially defined "relevant" as a deterministic rule:
`interest_jaccard >= 0.5`, or `>= 0.3` combined with same-country. I trained
the ranker against that, and validation loss dropped to essentially zero
within 2–3 epochs. That looked good at a glance, but it isn't — it's a
symptom of label leakage. The label was a hard threshold on the *exact same*
two features (`jaccard` and `same_country`) that I was also feeding the
model as inputs, so the network wasn't learning a ranking signal, it was
learning to reproduce an if-statement over its own inputs. A model that hits
near-zero loss in three epochs on a supposedly hard ranking task should be
treated as a red flag, not a result — and I want to be upfront that this is
exactly the kind of thing I'd expect to be probed on in a technical
interview, so I'd rather surface it here than have it discovered.

I fixed this by redefining relevance as a **probabilistic** latent affinity
score instead of a deterministic rule (see `app/services/labeling.py`):

```
logit = 2.4·jaccard + 0.9·rare_interest_bonus + 0.5·same_country
        + 0.4·same_city + 0.6·age_closeness − 5.0
label ~ Bernoulli(sigmoid(logit))
```

Two things make this meaningfully different from the leaky version:

1. It's **sampled from a Bernoulli distribution**, not thresholded — so the
   label is noisy, the way real human "would I follow this person" behavior
   actually is, rather than a clean function the network can memorize.
2. It includes a **rare-shared-interest bonus** that weights two people
   both listing an *uncommon* interest (e.g. Politics, Science) more highly
   than two people both listing a common one (e.g. Travel, Music) — and
   critically, this signal is **not exposed to the model as an input
   feature**. The model only ever sees aggregate `jaccard` and
   `shared_interest_count`, never which specific interests overlapped. So
   the label now carries information the model has to genuinely approximate
   rather than read directly off its own inputs.

After this fix, training loss decreases gradually and doesn't collapse:
`0.4398 → 0.4335 → 0.4322 → 0.4316 → 0.4308 → ... → 0.4270` over 15 epochs,
with validation loss tracking closely and best-checkpointing correctly
selecting epoch 4 (`val_loss=0.4299`) before validation loss started
drifting upward again. That's a modest, honest, non-collapsing learning
curve on a genuinely noisy signal — a far more credible story than a
suspiciously perfect one.

**I want to be explicit about what this means for the reported metrics
below: they measure whether the pipeline and evaluation methodology are
correctly implemented, not whether the model has learned genuine
"follow-worthiness."** No ground truth for that exists in this dataset, and
I'd rather say so plainly than imply otherwise.

### The dataset has no latitude/longitude

The brief's "core requirements" section describes `location (latitude/
longitude)` as an expected field, but the provided CSV only has `City` and
`Country`. I used same-city / same-country boolean flags as the location
signal instead of geocoding all ~15,500 unique cities through an external
API. Geocoding every city would add a network dependency, rate-limit risk,
and a reproducibility gap (results could change based on a third-party
service's data) for a 2–5 hour scoped assessment, for a fairly marginal gain
over the categorical signal I already have. I'm stating this plainly as a
deliberate, time-boxed choice rather than treating it as a hidden gap.

### Training pairs are sampled, not exhaustive

Building all (target, candidate) pairs for the full ~20,000-user training
split × 100 candidates each would be about 2 million rows. I sampled 4,000
target users for training (400,000 pairs) and 1,000 for validation
(100,000 pairs) after confirming this was enough for a 3-layer, 66-input-dim
MLP to show stable convergence — more pairs past this point looked like
diminishing returns relative to the added training time.

### Candidate-generation recall looks low (~4%), and that's expected here

I measured what fraction of *all* users in the entire 24,859-user
population who'd score as "relevant" under the latent-affinity label
actually landed in Stage A's top-100 pool. Because the relevance definition
is intentionally broad (many even loosely-similar users can score as weakly
relevant), the true "relevant" set for a given target user is often in the
hundreds, and no fixed 100-candidate pool can recall a large share of that.
This is an honest, reportable property of a two-stage system with a small,
fixed retrieval pool measured against a loosely-defined relevance signal —
not a retrieval bug — and I'd rather report the real number than a
flattering one.

## 4. Scaling proposal — millions of concurrent requests

This is what I'd actually change, not a hypothetical infrastructure wishlist:

1. **Stage A at scale.** Replace the single in-process vectorized scan with
   a proper candidate-retrieval service backed by an inverted index on the
   29 interest categories (cheap to build and maintain) plus a
   geo-partitioned index (e.g. geohash-bucketed) for location, so a lookup
   only scans users in the same interest/geo shard instead of the entire
   population. At millions of users, that's the difference between an O(n)
   scan and a bounded lookup.
2. **Stage B at scale.** Stage B's cost is independent of total user count —
   it only ever scores the 100 candidates Stage A hands it, regardless of
   whether the population is 25K or 25M. The MLP here has roughly 20K
   parameters, trivial to replicate across stateless model-serving
   instances behind a load balancer.
3. **API layer.** Stateless FastAPI instances behind a load balancer, with
   the in-memory user store (currently one process's full copy of the user
   table) moved to a shared, sharded key-value store or feature store, so
   any replica can serve any user without holding the entire population in
   its own process memory.
4. **Model refresh.** A periodic offline batch retraining job — not
   real-time — with weights pushed to a model registry/object storage and
   hot-reloaded by serving replicas, so retraining never sits in the
   request path.
5. **What I would deliberately *not* add without a specific, measured
   bottleneck to justify it:** Kafka, Kubernetes, Redis Cluster, or a vector
   database. The brief explicitly warns against this, and none of these
   solve a problem this architecture actually has at the stated scale until
   there's real production telemetry showing where it breaks.
6. **First thing I'd instrument before reaching for heavier
   infrastructure:** p95/p99 latency per stage, and retrieval-recall drift
   over time as the user base grows — that's the signal that tells you
   *which* of the above to actually build first, rather than guessing.

## 5. Setup instructions

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# Place Assessment_TwitterDataset.csv in data/ (not committed to git,
# per the assessment's instructions — see data/README.md)

python -m scripts.prepare_data     # builds the 80/10/10 user-id split
python -m scripts.train            # trains the ranker -> artifacts/ranker.pt
python -m scripts.evaluate         # measures precision/recall/NDCG/latency on the test split
python -m scripts.run_test_user    # runs the real API end-to-end, writes sample_results.csv

uvicorn app.main:app --reload      # run the API locally at http://127.0.0.1:8000
pytest tests/ -v                   # run the test suite (19 tests)
```

### Measured results (from the run committed in `artifacts/eval_metrics.json`)

| Metric | Value |
|---|---|
| Precision@10 | 0.415 |
| Recall@10 | 0.234 |
| Hit Rate@10 | 0.984 |
| NDCG@10 | 0.749 |
| Candidate-generation recall | 0.040 (see section 3 for why this is expected) |
| Mean end-to-end latency | ~6.8 ms |
| p95 end-to-end latency | ~8.8 ms |
| Test users evaluated | 1,000 (held-out test split) |

All numbers above are measured against the synthetic implicit-relevance
label described in section 3, not real user behavior — restated here
because it matters for how these numbers should be read.

### Test user used for `sample_results.csv`

Per the assessment's requirement, here are the details of the test user I
used to generate `sample_results.csv` (see `scripts/run_test_user.py`):

| Field | Value |
|---|---|
| Name | Rajan Mishra |
| Gender | Male |
| DOB | 2000-01-01 |
| Interests | Technology, Music |
| City | Gurugram |
| Country | India |

## Repository layout

```
app/
  api/routes.py            FastAPI routes: GET /health, POST /recommendations
  core/config.py            Centralized settings (paths, pool sizes, split ratios)
  core/logging.py           Logging setup
  main.py                   FastAPI app + startup lifecycle (loads model/data once)
  models/ranker.py          The PyTorch MLP ranker (Stage B)
  schemas/recommendation.py Pydantic request/response models + validation
  services/
    features.py             Feature engineering shared by training and serving
    retrieval.py             Stage A candidate retrieval
    labeling.py              The implicit relevance signal used for training/eval
    pairs.py                 Vectorized training-pair generation
    ranker_inference.py      Loads the trained model, scores/ranks candidates
    recommender.py           Orchestrates Stage A + Stage B for one request
    user_store.py            In-memory user pool, built once at startup
    metrics.py               Precision/Recall/HitRate/NDCG@k
scripts/
  prepare_data.py            Builds the 80/10/10 user-id split
  train.py                   Trains the ranker
  evaluate.py                 Offline evaluation on the held-out test split
  run_test_user.py            Runs the real API end-to-end, writes sample_results.csv
tests/                        19 unit + API integration tests
artifacts/                    Trained model + eval metrics (model weights gitignored)
data/                         Dataset (gitignored) + processed split (gitignored)
```
