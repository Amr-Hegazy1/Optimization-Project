# Router Failure Mode & Regret Analysis

Model: `runs/router/model.joblib`

Dataset: `runs/router/dataset_combined_min_cost.csv`

Rows: 9179

SA threshold stored in model: 0.47500000000000003

## Confusion matrix

| True\Pred | sa | ga | aco |
|---|---:|---:|---:|
| sa | 1055 | 21 | 2 |
| ga | 62 | 2989 | 50 |
| aco | 1 | 70 | 4929 |


## Classification report

```
              precision    recall  f1-score   support

         aco     0.9896    0.9858    0.9877      5000
          ga     0.9705    0.9639    0.9672      3101
          sa     0.9436    0.9787    0.9608      1078

    accuracy                         0.9776      9179
   macro avg     0.9679    0.9761    0.9719      9179
weighted avg     0.9777    0.9776    0.9776      9179

```

## Safety subsection: ACO ground truth misrouted to SA

Count(true=aco, pred=sa): **1 / 5000** (rate=0.000200).

Interpretation: if this count is 0 (or extremely low), the router is conservative about selecting SA when ACO is truly optimal by cost.

## Regret (cost overhead vs best-cost oracle)

Overall regret summary: mean=2.380281, median=0.000000, p90=0.000000, p99=12.167947, max=1695.411425.

### Regret by true label

| True label | Count | Mean | Median | P90 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| sa | 1078 | 1.440282 | 0.000000 | 0.000000 | 2.353816 | 707.501126 |
| ga | 3101 | 6.232662 | 0.000000 | 0.000000 | 113.671926 | 1695.411425 |
| aco | 5000 | 0.193698 | 0.000000 | 0.000000 | 2.955255 | 314.326823 |


### Safety regret slice (true=aco, pred=sa)

If any cases exist, regret summary: mean=70.941514, median=70.941514, p90=70.941514, p99=70.941514, max=70.941514.
