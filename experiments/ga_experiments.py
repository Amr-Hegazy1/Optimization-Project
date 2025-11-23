"""
Genetic Algorithm experiment runner mirroring the report's GA case studies.

Covers population/generation sweeps (C1-C6), weight tuning (C7-C9),
mutation/elitism sweeps (C10-C12), robot-count variations (C13-C14),
and operator combinations (C15-C18).
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
from experiments.utils import (
    ExperimentResult,
    apply_scenario,
    build_initial_movements,
    coverage_percentage,
)
from optimization.base_optimizer import BaseOptimizer
from genetic import GeneticOptimizer


# Keep experiments headless.
config.ENABLE_VISUALIZATION = False
config.FAST_MODE = True


GA_EXPERIMENTS: List[Dict[str, object]] = [
    # Population / generation sweeps (C1-C6)
    {
        "name": "C1_pop20_gen200",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C2_pop10_gen200",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 10,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C3_pop30_gen200",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 30,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C4_pop20_gen100",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 100,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C5_pop20_gen300",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 300,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C6_pop20_gen400",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 400,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },

    # Objective weight sweeps (C7-C9)
    {
        "name": "C7_alpha100_beta200",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 100,
        "beta": 200,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C8_alpha9000_beta7000",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 9000,
        "beta": 7000,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C9_alpha10000_beta670",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 10000,
        "beta": 670,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },

    # Mutation / elitism sweeps (C10-C12)
    {
        "name": "C10_mut10_elite10",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.1,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C11_mut10_elite40",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.1,
        "elite_rate": 0.4,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C12_mut40_elite10",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.4,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },

    # Robot-count variations (C13-C14)
    {
        "name": "C13_robots4_steps130",
        "num_robots": 4,
        "path_length": 130,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C14_robots8_steps90",
        "num_robots": 8,
        "path_length": 90,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point_per_robots_paths",
    },

    # Operator combinations (C15-C18)
    {
        "name": "C15_swap_one_point_roulette",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap",
        "parent_selection_method": "roulette",
        "crossover_method": "one_point",
    },
    {
        "name": "C16_swap_per_robot_one_point_sus",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "sus",
        "crossover_method": "one_point",
    },
    {
        "name": "C17_swap_one_point_per_robot_roulette",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap",
        "parent_selection_method": "roulette",
        "crossover_method": "one_point_per_robots_paths",
    },
    {
        "name": "C18_swap_per_robot_one_point_per_robot_roulette",
        "num_robots": 6,
        "path_length": 125,
        "alpha": 3000,
        "beta": 500,
        "gamma": 4.0,
        "population": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_rate": 0.1,
        "mutation_method": "swap_per_robot_path",
        "parent_selection_method": "roulette",
        "crossover_method": "one_point_per_robots_paths",
    },
]


def run_experiment(row: Dict[str, object]) -> ExperimentResult:
    """Execute a single GA configuration and return metrics."""
    positions = apply_scenario(
        num_robots=int(row["num_robots"]),
        path_length=int(row["path_length"]),
        alpha=float(row["alpha"]),
        beta=float(row["beta"]),
        gamma=float(row["gamma"]),
        energy_budget=int(row["path_length"]),
    )

    initial_movements = build_initial_movements(
        path_length=int(row["path_length"]),
        positions=positions,
    )

    optimizer = GeneticOptimizer(
        population_size=int(row["population"]),
        generation_size=int(row["generations"]),
        mutation_rate=float(row["mutation_rate"]),
        elite_rate=float(row["elite_rate"]),
        visualizer=None,
    )

    best_movements, best_cost = optimizer.run(
        initial_movements,
        mutation_method=str(row["mutation_method"]),
        parent_selection_method=str(row["parent_selection_method"]),
        crossover_method=str(row["crossover_method"]),
        robot_positions=positions,
    )
    best_path = BaseOptimizer.movements_to_positions(best_movements, positions)
    coverage = coverage_percentage(best_path)

    return ExperimentResult(
        name=str(row["name"]),
        cost=float(best_cost),
        coverage=coverage,
        iterations=int(row["generations"]),
        extra={
            "pop": row["population"],
            "mut_rate": row["mutation_rate"],
            "elite": row["elite_rate"],
            "mut": row["mutation_method"],
            "cross": row["crossover_method"],
            "select": row["parent_selection_method"],
            "robots": row["num_robots"],
            "steps": row["path_length"],
            "alpha": row["alpha"],
            "beta": row["beta"],
        },
    )


def main() -> None:
    results: List[ExperimentResult] = []
    print("\nRunning Genetic Algorithm experiments...\n")
    for row in GA_EXPERIMENTS:
        print(f"==> {row['name']}")
        result = run_experiment(row)
        results.append(result)
        print(result.as_row())
    print("\nAll GA experiments completed.\n")


if __name__ == "__main__":
    main()
