# Router Dataset Generation (SA/GA/ACO)

This document describes how we generated the training data used for the SA/GA/ACO router classifier in this repository (December 2025).

## Goal

Train a multiclass router that selects **Simulated Annealing (SA)**, **Genetic Algorithm (GA)**, or **Ant Colony Optimization (ACO)** for a given scenario.

### Label definition (best cost)

For each sampled scenario, we run SA, GA, and ACO under the **same budget** and assign the label as the algorithm with the **minimum achieved cost**:

- Label = $\arg\min_{a \in \{\mathrm{sa},\mathrm{ga},\mathrm{aco}\}} \mathrm{cost}(a)$

Ties are broken by selecting the faster algorithm (measured runtime in seconds).

## Data schema

Each row in the dataset contains:

- Feature columns: `f_<name>` for all names in `optimization/router/features.py` (`DEFAULT_FEATURE_SPEC`)
- Label column: `label` in `{sa, ga, aco}`
- Per-algorithm outcomes (recorded for analysis/relabeling):
  - `sa_cost`, `ga_cost`, `aco_cost`
  - `sa_sec`, `ga_sec`, `aco_sec`
- Experiment metadata: `budget`, `seed`

## Step 1 — Base dataset collection (broad sampling)

We first generated a “base” dataset by randomly sampling scenarios (robots/path length and optimizer hyperparameters), running all three algorithms, and labeling by best cost.

Command (example):

```bash
cd "/home/amr/Optimization Project"
source .venv/bin/activate

python ml_data_collection.py \
  --samples 20000 \
  --budget 50 \
  --workers 12 \
  --chunksize 1 \
  --quiet \
  --label-policy min_cost \
  --out runs/router/dataset.csv
```

Notes:

- `--quiet` suppresses verbose optimizer printing.
- Multiprocessing is used (`--workers > 1`) to parallelize CPU-bound optimizer runs.

## Step 2 — Targeted SA enrichment (fast SA-positive mining)

Under strict best-cost labeling, SA is often rare. To avoid collecting hundreds of thousands of fully random samples, we generated an **SA-enriched** dataset by biasing sampling toward a region where SA wins more frequently.

### Targeting strategy

From exploratory analysis on the base dataset, SA best-cost wins were more common in scenarios with:

- Longer paths (e.g., `path_length` in `[110, 150]`)
- Lower connectivity thresholds (e.g., `connectivity_threshold` in `[6, 9]`)
- Low initial connectivity (low `start_connectivity_edge_frac`)

We therefore used:

- Scenario range restrictions (`--min-path/--max-path`, `--connectivity-min/--connectivity-max`)
- A **cheap prefilter** (`--prefilter-edge-max`) based on feature extraction (no optimizer runs) to skip cases that are unlikely to produce SA wins.
- `--keep-only-label sa` to write only SA-best-cost rows.
- `--target-written N` to stop once we have enough SA wins.

Command used to generate 1000 SA-best-cost examples:

```bash
cd "/home/amr/Optimization Project"
source .venv/bin/activate

python ml_data_collection.py \
  --budget 50 \
  --workers 12 \
  --chunksize 4 \
  --quiet \
  --label-policy min_cost \
  --min-path 110 --max-path 150 \
  --connectivity-min 6 --connectivity-max 9 \
  --prefilter-edge-max 0.2 \
  --keep-only-label sa \
  --target-written 1000 \
  --max-attempts 100000 \
  --out runs/router/dataset_sa_min_cost_1k.csv
```

This produces a file containing only `label=sa` rows, but retains full cost/time columns for all three algorithms.

## Step 3 — Relabel base dataset to best-cost (sanity/alignment)

If the base dataset was previously labeled with another policy (e.g., Pareto/utility), it can be relabeled to strict best-cost using the stored cost/time columns.

Command:

```bash
cd "/home/amr/Optimization Project"
source .venv/bin/activate

python router_dataset_relabel.py \
  --in runs/router/dataset.csv \
  --out runs/router/dataset_min_cost.csv \
  --label-policy min_cost
```

## Step 4 — Combine datasets and control class imbalance

We then combined:

- Base best-cost dataset: `runs/router/dataset_min_cost.csv`
- SA-enriched dataset: `runs/router/dataset_sa_min_cost_1k.csv`

Because ACO dominates the base dataset under best-cost labeling, we capped the maximum number of samples per class during combination (to reduce extreme imbalance during supervised training).

Command:

```bash
cd "/home/amr/Optimization Project"
source .venv/bin/activate

python router_dataset_combine.py \
  --in runs/router/dataset_min_cost.csv \
  --in runs/router/dataset_sa_min_cost_1k.csv \
  --out runs/router/dataset_combined_min_cost.csv \
  --cap-per-class 5000 \
  --seed 0
```

Resulting (example) class counts after capping:

- SA: 1078
- GA: 3101
- ACO: 5000

## Step 5 — Train the router model

We train a classifier on the combined dataset and evaluate using a stratified train/test split.

Command:

```bash
cd "/home/amr/Optimization Project"
source .venv/bin/activate

python router_train.py \
  --data runs/router/dataset_combined_min_cost.csv \
  --class-weights \
  --seed 0
```

Notes:

- Training uses a RandomForest by default and may tune an SA probability threshold on a validation split to improve the SA precision/recall tradeoff.
- The trained model is saved to `runs/router/model.joblib`.

## Reproducibility

- Each scenario is generated from a deterministic seed derived from the global `--seed` and sample index.
- For multiprocessing, each worker uses its own deterministic per-sample seed so results are stable across process counts/scheduling (subject to non-determinism inside external libraries, if any).
