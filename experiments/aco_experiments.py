"""
Ant Colony Optimization experiment runner.

Explores the impact of pheromone/heuristic weights, evaporation, and strategy
choice (AS vs SACO) on path-planning performance.
"""

from __future__ import annotations

from typing import Dict, List
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from ant_colony import AntColonyOptimizer
from experiments.utils import (
    ExperimentResult,
    apply_scenario,
    build_initial_movements,
    coverage_percentage,
)
from optimization.base_optimizer import BaseOptimizer


# Keep runs fast and console-friendly.
config.ENABLE_VISUALIZATION = False
config.FAST_MODE = True


ACO_EXPERIMENTS: List[Dict[str, object]] = [
    {
        "name": "aco_as_baseline",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 0.5,
        "beta": 1.0,
        "evaporation": 0.4,
        "num_ants": 30,
        "iterations": 200,
        "strategy": "as",
    },
    {
        "name": "aco_as_high_beta",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 0.5,
        "beta": 2.0,
        "evaporation": 0.35,
        "num_ants": 30,
        "iterations": 200,
        "strategy": "as",
    },
    {
        "name": "aco_as_low_evap_high_alpha",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 1.0,
        "beta": 1.0,
        "evaporation": 0.2,
        "num_ants": 40,
        "iterations": 200,
        "strategy": "as",
    },
    {
        "name": "aco_saco_pheromone_only",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 0.7,
        "beta": 0.0,
        "evaporation": 0.4,
        "num_ants": 35,
        "iterations": 200,
        "strategy": "saco",
    },
    {
        "name": "aco_saco_fast_evap",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 0.7,
        "beta": 0.0,
        "evaporation": 0.6,
        "num_ants": 35,
        "iterations": 200,
        "strategy": "saco",
    },
]


def run_experiment(row: Dict[str, object]) -> ExperimentResult:
    """Execute one ACO configuration and report metrics."""
    positions = apply_scenario(
        num_robots=int(row["num_robots"]),
        path_length=int(row["path_length"]),
        alpha=3000,
        beta=500,
        gamma=4.0,
        energy_budget=int(row["path_length"]),
    )

    initial_movements = build_initial_movements(
        path_length=int(row["path_length"]),
        positions=positions,
    )

    optimizer = AntColonyOptimizer(
        num_ants=int(row["num_ants"]),
        max_iterations=int(row["iterations"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        evaporation_rate=float(row["evaporation"]),
        strategy_name=str(row["strategy"]),
        visualizer=None,
    )

    best_path = optimizer.run(initial_movements)
    best_cost = BaseOptimizer.cost_function(best_path)
    coverage = coverage_percentage(best_path)

    return ExperimentResult(
        name=str(row["name"]),
        cost=float(best_cost),
        coverage=coverage,
        iterations=int(row["iterations"]),
        extra={
            "ants": row["num_ants"],
            "alpha": row["alpha"],
            "beta": row["beta"],
            "evap": row["evaporation"],
            "strategy": row["strategy"],
        },
    )


def main() -> None:
    results: List[ExperimentResult] = []
    print("\nRunning Ant Colony Optimization experiments...\n")
    for row in ACO_EXPERIMENTS:
        print(f"==> {row['name']}")
        result = run_experiment(row)
        results.append(result)
        print(result.as_row())
    print("\nAll ACO experiments completed.\n")


if __name__ == "__main__":
    main()
