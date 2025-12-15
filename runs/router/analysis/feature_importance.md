# Router Feature Importance Analysis

Model: `runs/router/model.joblib`

Dataset: `runs/router/dataset_combined_min_cost.csv`

Rows: 9179

Label counts (dataset): {'aco': 5000, 'ga': 3101, 'sa': 1078}

SA threshold stored in model: 0.47500000000000003

## What this measures

We report (1) tree-based Gini importance when the classifier exposes it, and (2) permutation importance using macro-F1. Higher values mean the router’s performance drops more when that feature is shuffled.

## Global (multiclass) importance

### Permutation importance (macro-F1)

| Feature | Importance (mean) | Std |
|---|---:|---:|
| start_connectivity_edge_frac | 0.256844 | 0.005229 |
| connectivity_threshold | 0.244106 | 0.003882 |
| energy_budget | 0.046638 | 0.002468 |
| path_length | 0.039992 | 0.001982 |
| alpha | 0.018435 | 0.000830 |
| beta | 0.018111 | 0.001039 |
| communication_radius | 0.010214 | 0.000493 |
| gamma | 0.010120 | 0.000410 |
| n_robots | 0.006405 | 0.000664 |
| start_mean_pairwise_manhattan | 0.005165 | 0.000678 |
| start_mean_pairwise_euclidean | 0.005036 | 0.000701 |
| start_min_pairwise_euclidean | 0.003194 | 0.000723 |
| start_min_pairwise_manhattan | 0.002529 | 0.001044 |
| map_height | 0.000000 | 0.000000 |
| map_width | 0.000000 | 0.000000 |


### RF Gini importance (if available)

| Feature | Importance (mean) | Std |
|---|---:|---:|
| connectivity_threshold | 0.315229 | 0.000000 |
| start_connectivity_edge_frac | 0.273568 | 0.000000 |
| start_min_pairwise_euclidean | 0.067905 | 0.000000 |
| energy_budget | 0.067079 | 0.000000 |
| start_min_pairwise_manhattan | 0.061558 | 0.000000 |
| path_length | 0.057531 | 0.000000 |
| n_robots | 0.045963 | 0.000000 |
| beta | 0.023061 | 0.000000 |
| alpha | 0.022733 | 0.000000 |
| start_mean_pairwise_euclidean | 0.020976 | 0.000000 |
| communication_radius | 0.015921 | 0.000000 |
| gamma | 0.014435 | 0.000000 |
| start_mean_pairwise_manhattan | 0.014042 | 0.000000 |
| map_height | 0.000000 | 0.000000 |
| map_width | 0.000000 | 0.000000 |


## ACO vs SA (pairwise) importance

This section restricts evaluation to samples whose ground-truth label is either `aco` or `sa`, then computes permutation importance on that subset. This helps interpret which features drive the ACO-vs-SA boundary.

| Feature | Importance (mean) | Std |
|---|---:|---:|
| start_connectivity_edge_frac | 0.228198 | 0.002021 |
| connectivity_threshold | 0.034630 | 0.001305 |
| energy_budget | 0.012326 | 0.000696 |
| path_length | 0.011423 | 0.000699 |
| alpha | 0.005827 | 0.000170 |
| beta | 0.005657 | 0.000056 |
| gamma | 0.005127 | 0.000234 |
| communication_radius | 0.004797 | 0.000063 |
| n_robots | 0.003034 | 0.000251 |
| start_mean_pairwise_euclidean | 0.002971 | 0.000178 |
| start_mean_pairwise_manhattan | 0.002909 | 0.000446 |
| start_min_pairwise_manhattan | 0.001522 | 0.000415 |
| start_min_pairwise_euclidean | 0.001496 | 0.000417 |
| map_height | 0.000000 | 0.000000 |
| map_width | 0.000000 | 0.000000 |

