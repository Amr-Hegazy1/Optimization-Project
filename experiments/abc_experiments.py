"""
Artificial Bee Colony experiment runner mirroring the GA/ACO harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from abc_optimizer import ABCOptimizer
from experiments.utils import (
    ExperimentResult,
    apply_scenario,
    build_initial_movements,
    coverage_percentage,
    reset_defaults,
)
from optimization.base_optimizer import BaseOptimizer

# Keep experiments headless
config.ENABLE_VISUALIZATION = False
config.FAST_MODE = True


ABC_EXPERIMENTS: List[Dict[str, object]] = [
    {
        "name": "ABC_base",
        "num_robots": 6,
        "path_length": 100,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "colony_size": 24,
        "cycles": 200,
        "limit": 12,
        "onlooker_ratio": 0.5,
        "neighbor_window": 10,
        "neighbor_attempts": 5,
    },
    {
        "name": "ABC_large_colony",
        "num_robots": 6,
        "path_length": 100,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "colony_size": 32,
        "cycles": 200,
        "limit": 10,
        "onlooker_ratio": 0.6,
        "neighbor_window": 12,
        "neighbor_attempts": 6,
    },
]


def run_experiments() -> None:
    results: List[ExperimentResult] = []

    try:
        for exp in ABC_EXPERIMENTS:
            positions = apply_scenario(
                num_robots=exp["num_robots"],
                path_length=exp["path_length"],
                alpha=exp["alpha"],
                beta=exp["beta"],
                gamma=exp["gamma"],
            )

            initial_movements = build_initial_movements(
                path_length=exp["path_length"], positions=positions
            )

            optimizer = ABCOptimizer(
                colony_size=exp["colony_size"],
                max_cycles=exp["cycles"],
                limit=exp["limit"],
                onlooker_ratio=exp["onlooker_ratio"],
                neighbor_window=exp["neighbor_window"],
                neighbor_attempts=exp["neighbor_attempts"],
                visualizer=None,
            )

            best_movements, best_cost = optimizer.run(initial_movements)
            best_path = BaseOptimizer.movements_to_positions(best_movements, positions)
            cover = coverage_percentage(best_path)

            results.append(
                ExperimentResult(
                    name=exp["name"],
                    cost=best_cost,
                    coverage=cover,
                    iterations=exp["cycles"],
                    extra={
                        "colony": exp["colony_size"],
                        "limit": exp["limit"],
                        "onlookers": exp["onlooker_ratio"],
                    },
                )
            )
    finally:
        reset_defaults()

    for res in results:
        print(res.as_row())


if __name__ == "__main__":
    run_experiments()
