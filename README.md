# Floter AI - Two-Stage Recommendation Engine

A two-stage "who to follow" recommendation backend. A fast in-memory
candidate retrieval stage narrows the full user pool down to 100
candidates, then a PyTorch MLP ranker scores and re-ranks those 100 down
to a final top 10. Served behind a FastAPI endpoint.

This README explains what was built and why. The brief was explicit that
a simple, well-understood, properly justified solution should score
higher than an over-engineered one that can't be defended, and that is
the standard this was held to.

## 1. AI and external code attribution

I want to be direct about how this was built, since the brief asks for
that explicitly and I would rather be over-clear than have it come up as
a surprise in the interview.

I used Claude (Anthropic's AI coding assistant) as the primary implementer
for this project: writing the retrieval, ranking and training code,
structuring the repo, and running the training and evaluation scripts. My
own role was to make every architectural decision, set the direction at
each step, review what was produced, and verify the actual results rather
than take them on faith. Concretely: I chose the MLP over a Two-Tower
architecture and can explain why (section 2), I chose the retrieval
scoring weights and why they're ordered the way they are (section 2), and
when the first version of the training label converged suspiciously fast,
I treated that as a red flag rather than a result, required it to be
re-examined, and the fix that followed is documented in section 3. I have
read through every file in this repo, understand what each one does, and
can walk through and justify the implementation in the technical
interview, which is the standard the brief holds this to.

I'm stating this plainly rather than writing this document as if I typed
every line myself, because I think that would be a worse answer to "who
wrote this" than the true one.

To directly answer the question the brief asks about this specific
document: the brief requires the README to be human-written, and I want
to be precise about what that meant in practice here rather than leave it
ambiguous. The reasoning in this README, the decisions it explains, and
the numbers it cites are mine, checked against the actual code and the
actual files in artifacts/. But I drafted the sentences with Claude's
help rather than typing every line from a blank page, the same way I used
it for the code. If asked directly in the interview whether I typed this
document character by character myself, the honest answer is no, and I'd
rather say that here than have the answer feel like it contradicts this
section when it's asked out loud.

No third-party recommendation library, pretrained model or vector
database was used. The only dependencies are standard, widely used
open-source libraries: PyTorch, FastAPI, pandas, NumPy, scikit-learn and
Pydantic, all listed with pinned versions in requirements.txt.

## 2. Design decisions

### Stage A: candidate retrieval (app/services/retrieval.py)

The decision here was a single vectorized weighted-sum scan over the
whole user pool instead of building an inverted interest index or an ANN
structure.

With around 25,000 users, a full NumPy pass (logical_and/logical_or over a
24,859 by 29 boolean interest matrix, plus a few array comparisons for
location and age) runs in single digit milliseconds. Measured mean latency
for the entire retrieval-plus-ranking pipeline is under 10ms (see section
5). An inverted index or an ANN library would add real complexity: build
time, memory overhead, staleness on updates, for no measurable latency win
at this scale. That's exactly the kind of over-engineering the brief warns
against. If this dataset were 10 to 100 million users instead of 25
thousand, this would be the first thing to replace, and section 4
describes what it would be replaced with.

The retrieval score is:

score = 2.0 x jaccard(interests) + 1.0 x same_country + 1.5 x same_city + 0.5 x age_score

Interest overlap is weighted highest because it is the strongest, most
information dense signal available in this dataset. The 29 category
interest vocabulary is fixed and every user has it populated. Same-city
is weighted above same-country because a city match is rare and
meaningful (the dataset has 222 countries but roughly 15,500 distinct
cities), while a country match is common and therefore a weaker signal
on its own.

### Stage B: ranking (app/models/ranker.py)

The decision here was a pairwise-feature MLP over a Two-Tower
architecture, which the brief explicitly offered as an option.

A Two-Tower model's main advantage is precomputing item embeddings once
and doing fast approximate nearest-neighbor lookup over a huge item
catalog. That advantage does not apply here, because Stage A has already
narrowed the candidate set to 100 before Stage B ever runs. Stage B only
ever needs to score 100 pairs, not search millions. A Two-Tower model
would also mean the network learns opaque embeddings instead of the
hand-interpretable features (interest Jaccard, same city, age gap) I can
name and defend individually. Given the brief's stated preference for
justified simplicity, the pairwise MLP was the better choice for this
problem at this scale.

Architecture: a 3-layer MLP, Linear(66 to 64), ReLU, Linear(64 to 32),
ReLU, Linear(32 to 1). Trained with BCEWithLogitsLoss, Adam optimizer at
lr=1e-3, batch size 64, 15 epochs, keeping the best-validation-loss
checkpoint.

Input (66 dimensions): the target user's feature vector (normalized age
plus a 29-dim interest multi-hot vector, 30 dims total), the candidate's
identical 30-dim vector, and 6 hand-engineered pairwise features: interest
Jaccard similarity, shared-interest count, same city, same country, same
gender, and age difference.

Hand-engineered multi-hot and Jaccard features were used rather than
learned embeddings for interests because the dataset has a small, fixed,
29 category controlled vocabulary, not free text. Learned embeddings earn
their complexity with thousands of noisy free-text tags. With 29 fixed
categories, a multi-hot vector and Jaccard similarity are the more direct,
interpretable representation, so that is what this uses.

## 3. Trade-offs and constraints

The two biggest limitations in this project are worth stating directly,
because hiding them would be worse than explaining them plainly.

### There is no real follow-graph in the dataset

Assessment_TwitterDataset.csv has no ground-truth "who follows whom", just
UserID, Name, Gender, DOB, Interests, City, Country. That means there is
no real label to train Stage B against, and this is genuinely the central
constraint of the whole project.

The first version of the training label was wrong, and it is worth
explaining why, because how it was caught and fixed is more informative
than pretending it did not happen. The first approach defined "relevant"
as a deterministic rule: interest_jaccard >= 0.5, or >= 0.3 combined with
same country. The ranker was trained against that, and validation loss
dropped to essentially zero within two or three epochs. That looked good
at a glance, but it was not. It was a symptom of label leakage. The label
was a hard threshold on the exact same two features (jaccard and
same_country) that were also being fed to the model as inputs, so the
network was not learning a ranking signal, it was learning to reproduce
an if-statement over its own inputs. A model that hits near-zero loss in
three epochs on a supposedly hard ranking task should be treated as a red
flag, not a result, and I caught this by reviewing the loss curve rather
than accepting the first number that came out. This is exactly the kind
of thing that should come up in a technical interview, so it is surfaced
here directly rather than left to be discovered.

The fix was to redefine relevance as a probabilistic latent affinity
score instead of a deterministic rule (see app/services/labeling.py):

logit = 2.4 x jaccard + 0.9 x rare_interest_bonus + 0.5 x same_country + 0.4 x same_city + 0.6 x age_closeness - 5.0
label is sampled from Bernoulli(sigmoid(logit))

Two things make this meaningfully different from the leaky version. First,
it is sampled from a Bernoulli distribution rather than thresholded, so
the label is noisy the way real human "would I follow this person"
behavior actually is, rather than a clean function the network can
memorize. Second, it includes a rare-shared-interest bonus that weights
two people who both list an uncommon interest, such as Politics or
Science, more highly than two people who both list a common one, such as
Travel or Music. Critically, this signal is not exposed to the model as an
input feature. The model only ever sees aggregate jaccard and
shared_interest_count, never which specific interests overlapped. So the
label now carries information the model has to genuinely approximate
rather than read directly off its own inputs.

After this fix, training loss decreases gradually and does not collapse:
0.4398, 0.4335, 0.4322, 0.4316, 0.4308, and so on down to 0.4270 over 15
epochs, with validation loss tracking closely and best-checkpointing
correctly selecting epoch 4 (val_loss = 0.4299) before validation loss
started drifting upward again. That is a modest, honest, non-collapsing
learning curve on a genuinely noisy signal, which is a far more credible
result than a suspiciously perfect one.

It is worth being explicit about what this means for the reported metrics
below. They measure whether the pipeline and evaluation methodology are
correctly implemented, not whether the model has learned genuine
"follow-worthiness." No ground truth for that exists in this dataset,
and that should be stated plainly rather than implied otherwise.

### The dataset has no latitude and longitude

The brief's core requirements section describes location as latitude and
longitude, but the provided CSV only has City and Country. I used
same-city and same-country boolean flags as the location signal instead
of geocoding all roughly 15,500 unique cities through an external API.
Geocoding every city would add a network dependency, rate-limit risk, and
a reproducibility gap (results could change based on a third-party
service's data) for a 2 to 5 hour scoped assessment, for a fairly marginal
gain over the categorical signal I already had. I am stating this plainly
as a deliberate, time-boxed choice rather than treating it as a hidden
gap.

### Training pairs are sampled, not exhaustive

Building all target and candidate pairs for the full roughly 20,000-user
training split times 100 candidates each would be about 2 million rows. I
sampled 4,000 target users for training (400,000 pairs) and 1,000 for
validation (100,000 pairs) after confirming this was enough for a 3-layer,
66-input-dimension MLP to show stable convergence. More pairs past this
point looked like diminishing returns relative to the added training time.

### Candidate-generation recall looks low (around 4 percent), and that is expected here

I measured what fraction of all users in the entire 24,859-user population
who would score as "relevant" under the latent-affinity label actually
landed in Stage A's top-100 pool. Because the relevance definition is
intentionally broad (many even loosely similar users can score as weakly
relevant), the true "relevant" set for a given target user is often in the
hundreds, and no fixed 100-candidate pool can recall a large share of
that. This is an honest, reportable property of a two-stage system with a
small, fixed retrieval pool measured against a loosely defined relevance
signal, not a retrieval bug. I would rather report the real number than a
flattering one.

## 4. Scaling proposal for millions of concurrent requests

This is what I would actually change, not a hypothetical infrastructure
wishlist.

Stage A at scale: replace the single in-process vectorized scan with a
proper candidate-retrieval service backed by an inverted index on the 29
interest categories, which is cheap to build and maintain, plus a
geo-partitioned index (for example geohash-bucketed) for location, so a
lookup only scans users in the same interest and geo shard instead of the
entire population. At millions of users, that is the difference between
an O(n) scan and a bounded lookup.

Stage B at scale: Stage B's cost is independent of total user count. It
only ever scores the 100 candidates Stage A hands it, whether the
population is 25 thousand or 25 million. The MLP here has 6,401
parameters (verified directly with sum(p.numel() for p in
PairwiseRanker().parameters())), trivial to replicate across stateless
model-serving instances behind a load balancer.

API layer: stateless FastAPI instances behind a load balancer, with the
in-memory user store (currently one process's full copy of the user
table) moved to a shared, sharded key-value store or feature store, so
any replica can serve any user without holding the entire population in
its own process memory.

Model refresh: a periodic offline batch retraining job, not real-time,
with weights pushed to a model registry or object storage and hot-loaded
by serving replicas, so retraining never sits in the request path.

What I would deliberately not add without a specific, measured bottleneck
to justify it: Kafka, Kubernetes, Redis Cluster, or a vector database. The
brief explicitly warns against this, and none of these solve a problem
this architecture actually has at the stated scale until there is real
production telemetry showing where it breaks.

The first thing I would instrument before reaching for heavier
infrastructure is p95 and p99 latency per stage, and retrieval-recall
drift over time as the user base grows. That is the signal that tells you
which of the above to actually build first, rather than guessing.

## 5. Setup instructions

    python -m venv .venv
    .venv\Scripts\activate            (Windows)
    source .venv/bin/activate         (macOS or Linux)

    pip install -r requirements.txt

    Place Assessment_TwitterDataset.csv in data/ (not committed to git,
    per the assessment instructions, see data/README.md)

    python -m scripts.prepare_data     builds the 80/10/10 user-id split
    python -m scripts.train            trains the ranker, saves artifacts/ranker.pt
    python -m scripts.evaluate         measures precision, recall, NDCG and latency on the test split
    python -m scripts.run_test_user    runs the real API end to end, writes sample_results.csv

    uvicorn app.main:app --reload      runs the API locally at http://127.0.0.1:8000
    pytest tests/ -v                   runs the test suite (19 tests)

### Measured results (from the run committed in artifacts/eval_metrics.json)

Precision@10: 0.415
Recall@10: 0.234
Hit Rate@10: 0.984
NDCG@10: 0.749
Candidate-generation recall: 0.040 (see section 3 for why this is expected)
Mean end-to-end latency: about 6.8 ms
p95 end-to-end latency: about 8.8 ms
Test users evaluated: 1,000 (held-out test split)

All numbers above are measured against the synthetic implicit-relevance
label described in section 3, not real user behavior. I am restating that
here because it matters for how these numbers should be read.

### Test user used for sample_results.csv

Per the assessment's requirement, here are the details of the test user I
used to generate sample_results.csv (see scripts/run_test_user.py):

Name: Rajan Mishra
Gender: Male
DOB: 2000-01-01
Interests: Technology, Music
City: Gurugram
Country: India

## Repository layout

    app/
      api/routes.py             FastAPI routes: GET /health, POST /recommendations
      core/config.py            Centralized settings (paths, pool sizes, split ratios)
      core/logging.py           Logging setup
      main.py                   FastAPI app and startup lifecycle (loads model and data once)
      models/ranker.py          The PyTorch MLP ranker (Stage B)
      schemas/recommendation.py Pydantic request/response models and validation
      services/
        features.py             Feature engineering shared by training and serving
        retrieval.py            Stage A candidate retrieval
        labeling.py             The implicit relevance signal used for training and evaluation
        pairs.py                Vectorized training-pair generation
        ranker_inference.py     Loads the trained model, scores and ranks candidates
        recommender.py          Orchestrates Stage A and Stage B for one request
        user_store.py           In-memory user pool, built once at startup
        metrics.py              Precision, Recall, HitRate and NDCG at k
    scripts/
      prepare_data.py           Builds the 80/10/10 user-id split
      train.py                  Trains the ranker
      evaluate.py               Offline evaluation on the held-out test split
      run_test_user.py          Runs the real API end to end, writes sample_results.csv
    tests/                      19 unit and API integration tests
    artifacts/                  Trained model and eval metrics (model weights gitignored)
    data/                       Dataset (gitignored) and processed split (gitignored)
