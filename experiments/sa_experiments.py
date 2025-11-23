"""
Standalone Simulated Annealing experiment runner.

Covers the SA scenarios discussed in the report (baseline C1-like setup,
cooling schedule variants, and the 10-robot validation case).
"""

from __future__ import annotations

from typing import Dict, List
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import config
from experiments.utils import (
    ExperimentResult,
    apply_scenario,
    build_initial_movements,
    coverage_percentage,
)
from optimization.base_optimizer import BaseOptimizer
from simulated_annealing import SimulatedAnnealing


# Disable visualization to keep experiments fast/CLI-friendly.
config.ENABLE_VISUALIZATION = False
config.FAST_MODE = True


SA_EXPERIMENTS: List[Dict[str, object]] = [
    {
        "name": "sa_c1_baseline",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "initial_temp": 10.0,
        "cooling_rate": 0.995,
        "min_temp": 0.1,
        "iterations": 200,
    },
    {
        "name": "sa_fast_cooling",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "initial_temp": 8.0,
        "cooling_rate": 0.990,
        "min_temp": 0.1,
        "iterations": 200,
    },
    {
        "name": "sa_slow_cooling",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "initial_temp": 12.0,
        "cooling_rate": 0.999,
        "min_temp": 0.05,
        "iterations": 250,
    },
    {
        "name": "sa_validation_10robots",
        "num_robots": 10,
        "path_length": 72,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "initial_temp": 10.0,
        "cooling_rate": 0.995,
        "min_temp": 0.1,
        "iterations": 200,
    },
]


def run_experiment(config_row: Dict[str, object]) -> ExperimentResult:
    """Run a single SA configuration and return its metrics."""
    positions = apply_scenario(
        num_robots=int(config_row["num_robots"]),
        path_length=int(config_row["path_length"]),
        alpha=float(config_row["alpha"]),
        beta=float(config_row["beta"]),
        gamma=float(config_row["gamma"]),
        energy_budget=int(config_row["path_length"]),
    )

    initial_movements = build_initial_movements(
        path_length=int(config_row["path_length"]),
        positions=positions,
    )

    optimizer = SimulatedAnnealing(
        initial_temperature=float(config_row["initial_temp"]),
        cooling_rate=float(config_row["cooling_rate"]),
        min_temperature=float(config_row["min_temp"]),
        max_iterations=int(config_row["iterations"]),
        visualizer=None,
    )

    best_movements, best_cost = optimizer.run(initial_movements)
    best_path = BaseOptimizer.movements_to_positions(best_movements, positions)
    coverage = coverage_percentage(best_path)

    return ExperimentResult(
        name=str(config_row["name"]),
        cost=float(best_cost),
        coverage=coverage,
        iterations=int(config_row["iterations"]),
        extra={
            "init_temp": config_row["initial_temp"],
            "cool": config_row["cooling_rate"],
            "min_temp": config_row["min_temp"],
            "robots": config_row["num_robots"],
            "steps": config_row["path_length"],
        },
    )


def main() -> None:
    results: List[ExperimentResult] = []
    print("\nRunning Simulated Annealing experiments...\n")
    for row in SA_EXPERIMENTS:
        print(f"==> {row['name']}")
        result = run_experiment(row)
        results.append(result)
        print(result.as_row())
    print("\nAll SA experiments completed.\n")


if __name__ == "__main__":
    main()
