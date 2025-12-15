from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


KEYS = ("sa", "ga", "aco")


@dataclass(frozen=True)
class RowScores:
    costs: Dict[str, float]
    secs: Dict[str, float]


def _parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as e:
        raise ValueError(f"Missing/invalid '{key}' in input CSV") from e


def _get_scores(row: Dict[str, str]) -> RowScores:
    costs = {k: _parse_float(row, f"{k}_cost") for k in KEYS}
    secs = {k: _parse_float(row, f"{k}_sec") for k in KEYS}
    return RowScores(costs=costs, secs=secs)


def _choose_label_min_cost(scores: RowScores) -> str:
    best_cost = min(scores.costs.values())
    candidates = [k for k in KEYS if abs(scores.costs[k] - best_cost) < 1e-12]
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda k: scores.secs[k])


def _choose_label_pareto(scores: RowScores, *, rel_cost_tol: float) -> str:
    best_cost = min(scores.costs.values())
    threshold = best_cost * (1.0 + float(rel_cost_tol))
    feasible = [k for k in KEYS if scores.costs[k] <= threshold]
    if not feasible:
        return _choose_label_min_cost(scores)
    return min(feasible, key=lambda k: scores.secs[k])


def _choose_label_utility(scores: RowScores, *, time_weight: float) -> str:
    w = float(time_weight)
    return min(KEYS, key=lambda k: (scores.costs[k] + w * scores.secs[k]))


def _relabel_rows(
    rows: List[Dict[str, str]],
    *,
    policy: str,
    pareto_tol: float,
    time_weight: float,
) -> None:
    policy = str(policy).lower().strip()
    for row in rows:
        scores = _get_scores(row)
        if policy == "pareto":
            row["label"] = _choose_label_pareto(scores, rel_cost_tol=pareto_tol)
        elif policy == "utility":
            row["label"] = _choose_label_utility(scores, time_weight=time_weight)
        else:
            row["label"] = _choose_label_min_cost(scores)


def _downsample(rows: List[Dict[str, str]], *, max_per_class: int, seed: int) -> List[Dict[str, str]]:
    if max_per_class <= 0:
        return rows

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("label", ""))].append(row)

    rng = random.Random(int(seed))
    out: List[Dict[str, str]] = []
    for k in KEYS:
        group = groups.get(k, [])
        if len(group) > max_per_class:
            group = rng.sample(group, k=max_per_class)
        out.extend(group)

    # Preserve any unexpected labels as-is (shouldn't exist but keep safe)
    for k, group in groups.items():
        if k in KEYS:
            continue
        out.extend(group)

    rng.shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Relabel an existing router dataset using stored cost/time columns.")
    p.add_argument("--in", dest="inp", required=True, help="Input CSV (must contain *_cost and *_sec columns).")
    p.add_argument("--out", dest="out", required=True, help="Output CSV path.")
    p.add_argument(
        "--label-policy",
        type=str,
        default="min_cost",
        choices=["min_cost", "pareto", "utility"],
        help="Relabeling policy.",
    )
    p.add_argument("--pareto-tol", type=float, default=0.05)
    p.add_argument("--time-weight", type=float, default=0.0)
    p.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="If set, downsample each of sa/ga/aco to at most this many rows after relabeling.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    with inp.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise SystemExit("Input CSV has no header")
        rows = list(reader)

    if "label" not in fieldnames:
        fieldnames.append("label")

    _relabel_rows(rows, policy=args.label_policy, pareto_tol=float(args.pareto_tol), time_weight=float(args.time_weight))
    rows = _downsample(rows, max_per_class=int(args.max_per_class), seed=int(args.seed))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Print summary
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[str(r.get("label", ""))] += 1
    print(f"Wrote {len(rows)} rows to {out}")
    print("Label counts:", {k: counts.get(k, 0) for k in KEYS})


if __name__ == "__main__":
    main()
