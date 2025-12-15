from __future__ import annotations

import argparse
import csv
import contextlib
import functools
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from ant_colony import AntColonyOptimizer
from genetic import GeneticOptimizer
from simulated_annealing import SimulatedAnnealing
from optimization.base_optimizer import BaseOptimizer
from optimization.router.features import DEFAULT_FEATURE_SPEC, extract_features, vectorize_features
from experiments.utils import apply_scenario, build_initial_movements, generate_positions

Position = Tuple[int, int]


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _deterministic_base_seed(global_seed: int, sample_index: int) -> int:
    # Deterministic across process counts / scheduling.
    # Keep within 32-bit for libraries that expect that.
    return int((global_seed * 1_000_003 + sample_index * 9_973) % (2**31 - 1))


@dataclass
class AlgoResult:
    cost: float
    seconds: float


def _run_sa(initial_movements: np.ndarray, max_iterations: int) -> AlgoResult:
    optimizer = SimulatedAnnealing(
        initial_temperature=config.SA_INITIAL_TEMPERATURE,
        cooling_rate=config.SA_COOLING_RATE,
        min_temperature=config.SA_MIN_TEMPERATURE,
        max_iterations=max_iterations,
        visualizer=None,
    )
    t0 = time.perf_counter()
    best_movements, best_cost = optimizer.run(initial_movements)
    _ = BaseOptimizer.movements_to_positions(best_movements, config.ROBOT_INITIAL_POSITIONS)
    return AlgoResult(cost=float(best_cost), seconds=time.perf_counter() - t0)


def _run_ga(initial_movements: np.ndarray, generation_size: int) -> AlgoResult:
    optimizer = GeneticOptimizer(
        population_size=config.GA_POPULATION_SIZE,
        generation_size=generation_size,
        mutation_rate=config.GA_MUTATION_RATE,
        elite_rate=config.GA_ELITE_RATE,
        visualizer=None,
    )
    t0 = time.perf_counter()
    _best_movements, best_cost = optimizer.run(
        initial_movements,
        mutation_method=config.GA_MUTATION_METHOD,
        parent_selection_method=config.GA_PARENT_SELECTION_METHOD,
        crossover_method=config.GA_CROSSOVER_METHOD,
        robot_positions=config.ROBOT_INITIAL_POSITIONS,
    )
    return AlgoResult(cost=float(best_cost), seconds=time.perf_counter() - t0)


def _run_aco(max_iterations: int) -> AlgoResult:
    optimizer = AntColonyOptimizer(
        num_ants=config.ACO_NUM_ANTS,
        max_iterations=max_iterations,
        alpha=config.ACO_ALPHA,
        beta=config.ACO_BETA,
        evaporation_rate=config.ACO_EVAPORATION_RATE,
        strategy_name=config.ACO_STRATEGY,
        visualizer=None,
    )
    t0 = time.perf_counter()
    best_path = optimizer.run(None)
    best_cost = BaseOptimizer.cost_function(best_path)
    return AlgoResult(cost=float(best_cost), seconds=time.perf_counter() - t0)


def _choose_label(results: Dict[str, AlgoResult]) -> str:
    # Primary: minimal cost. Tie-break: fastest.
    items = list(results.items())
    best_cost = min(v.cost for _, v in items)
    candidates = [(k, v) for k, v in items if abs(v.cost - best_cost) < 1e-9]
    if len(candidates) == 1:
        return candidates[0][0]
    return min(candidates, key=lambda kv: kv[1].seconds)[0]


def _choose_label_pareto(results: Dict[str, AlgoResult], *, rel_cost_tol: float) -> str:
    """Pick fastest algorithm within (1+tol) of best cost."""
    best_cost = min(v.cost for v in results.values())
    threshold = best_cost * (1.0 + float(rel_cost_tol))
    feasible = [(k, v) for k, v in results.items() if v.cost <= threshold]
    if not feasible:
        return _choose_label(results)
    return min(feasible, key=lambda kv: kv[1].seconds)[0]


def _choose_label_utility(results: Dict[str, AlgoResult], *, time_weight: float) -> str:
    """Pick argmin(cost + time_weight * seconds)."""
    w = float(time_weight)
    return min(results.items(), key=lambda kv: (kv[1].cost + w * kv[1].seconds))[0]
def _sample_scenario(
    rng: np.random.Generator,
    max_robots: int,
    min_path: int,
    max_path: int,
    *,
    communication_min: float,
    communication_max: float,
    connectivity_min: float,
    connectivity_max: float,
) -> Dict[str, object]:
    n_robots = int(rng.integers(2, max_robots + 1))
    path_length = int(rng.integers(min_path, max_path + 1))

    # Log-uniform sampling tends to be more stable for weights.
    alpha = float(10 ** rng.uniform(3.0, 4.2))
    beta = float(10 ** rng.uniform(2.0, 3.2))
    gamma = float(rng.uniform(0.5, 10.0))

    communication_radius = float(rng.uniform(float(communication_min), float(communication_max)))
    connectivity_threshold = float(rng.uniform(float(connectivity_min), float(connectivity_max)))

    positions = generate_positions(n_robots)
    return {
        "n_robots": n_robots,
        "path_length": path_length,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "energy_budget": path_length,
        "communication_radius": communication_radius,
        "connectivity_threshold": connectivity_threshold,
        "robot_positions": positions,
    }


def _collect_one(
    sample_index: int,
    *,
    global_seed: int,
    budget: int,
    max_robots: int,
    min_path: int,
    max_path: int,
    quiet: bool,
    label_policy: str,
    pareto_tol: float,
    time_weight: float,
    prefilter_edge_max: Optional[float],
    keep_only_label: Optional[str],
    communication_min: float,
    communication_max: float,
    connectivity_min: float,
    connectivity_max: float,
) -> Optional[Dict[str, object]]:
    # All heavy work happens in the worker process.
    # NOTE: config is process-local, so global mutation is safe here.
    config.ENABLE_VISUALIZATION = False

    base_seed = _deterministic_base_seed(global_seed, sample_index)
    rng = np.random.default_rng(base_seed)
    scenario = _sample_scenario(
        rng,
        max_robots,
        min_path,
        max_path,
        communication_min=float(communication_min),
        communication_max=float(communication_max),
        connectivity_min=float(connectivity_min),
        connectivity_max=float(connectivity_max),
    )

    apply_scenario(
        num_robots=int(scenario["n_robots"]),
        path_length=int(scenario["path_length"]),
        alpha=float(scenario["alpha"]),
        beta=float(scenario["beta"]),
        gamma=float(scenario["gamma"]),
        energy_budget=int(scenario["energy_budget"]),
        communication_radius=float(scenario["communication_radius"]),
        connectivity_threshold=float(scenario["connectivity_threshold"]),
        robot_positions=scenario["robot_positions"],
    )

    feats = extract_features(
        map_grid=np.array(config.MAP),
        robot_positions=tuple(config.ROBOT_INITIAL_POSITIONS),
        n_robots=int(config.R),
        path_length=int(config.PATH_LENGTH),
        alpha=float(config.ALPHA),
        beta=float(config.BETA),
        gamma=float(config.GAMMA),
        energy_budget=float(config.ENERGY_BUDGET),
        communication_radius=float(config.COMMUNICATION_RADIUS),
        connectivity_threshold=float(config.CONNECTIVITY_THRESHOLD),
    )
    feat_vec = vectorize_features(feats)

    # Cheap prefilter: reject scenarios unlikely to produce the requested label (e.g., SA wins)
    # before running expensive optimizers.
    if prefilter_edge_max is not None:
        if float(feats.get("start_connectivity_edge_frac", 1.0)) > float(prefilter_edge_max):
            return None

    results: Dict[str, AlgoResult] = {}
    sink = open(os.devnull, "w") if quiet else None
    suppress = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    suppress_err = contextlib.redirect_stderr(sink) if quiet else contextlib.nullcontext()
    try:
        with suppress, suppress_err:
            _set_all_seeds(base_seed)
            # Also suppress prints from initial-solution generation.
            initial_movements = build_initial_movements(config.PATH_LENGTH, config.ROBOT_INITIAL_POSITIONS)

            _set_all_seeds(base_seed + 1)
            results["sa"] = _run_sa(initial_movements, budget)

            _set_all_seeds(base_seed + 2)
            results["ga"] = _run_ga(initial_movements, budget)

            _set_all_seeds(base_seed + 3)
            results["aco"] = _run_aco(budget)
    except Exception:
        return None
    finally:
        if sink is not None:
            sink.close()

    if not any(np.isfinite(r.cost) for r in results.values()):
        return None

    policy = str(label_policy).lower().strip()
    if policy == "pareto":
        label = _choose_label_pareto(results, rel_cost_tol=float(pareto_tol))
    elif policy == "utility":
        label = _choose_label_utility(results, time_weight=float(time_weight))
    else:
        label = _choose_label(results)

    if keep_only_label is not None and str(label) != str(keep_only_label):
        return None
    feature_names = DEFAULT_FEATURE_SPEC.names
    row: Dict[str, object] = {}
    for name, value in zip(feature_names, feat_vec.tolist()):
        row[f"f_{name}"] = float(value)
    row.update(
        {
            "label": label,
            "sa_cost": results["sa"].cost,
            "ga_cost": results["ga"].cost,
            "aco_cost": results["aco"].cost,
            "sa_sec": results["sa"].seconds,
            "ga_sec": results["ga"].seconds,
            "aco_sec": results["aco"].seconds,
            "budget": int(budget),
            "seed": int(base_seed),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect training data for the SA/GA/ACO router.")
    parser.add_argument("--samples", type=int, default=200, help="Number of instances to generate.")
    parser.add_argument("--budget", type=int, default=150, help="Iteration/generation budget for each algorithm.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--max-robots", type=int, default=len(config.ROBOT_INITIAL_POSITIONS))
    parser.add_argument("--min-path", type=int, default=30)
    parser.add_argument("--max-path", type=int, default=150)
    parser.add_argument("--communication-min", type=float, default=6.0)
    parser.add_argument("--communication-max", type=float, default=25.0)
    parser.add_argument("--connectivity-min", type=float, default=6.0)
    parser.add_argument("--connectivity-max", type=float, default=25.0)
    parser.add_argument("--out", type=str, default="runs/router/dataset.csv")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of worker processes (use 1 to disable parallelism).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Multiprocessing chunksize (increase to reduce overhead).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose optimizer prints (recommended for large sample counts).",
    )
    parser.add_argument(
        "--label-policy",
        type=str,
        default="pareto",
        choices=["min_cost", "pareto", "utility"],
        help="How to label the best algorithm (avoids always labeling ACO when it wins on cost but is slow).",
    )
    parser.add_argument(
        "--pareto-tol",
        type=float,
        default=0.05,
        help="Relative cost tolerance for pareto policy (fastest within (1+tol)*best_cost).",
    )
    parser.add_argument(
        "--time-weight",
        type=float,
        default=0.0,
        help="Time penalty weight for utility policy: minimize cost + time_weight * seconds.",
    )
    parser.add_argument(
        "--prefilter-edge-max",
        type=float,
        default=None,
        help="If set, skip scenarios whose start_connectivity_edge_frac exceeds this value (cheap prefilter before running optimizers).",
    )
    parser.add_argument(
        "--keep-only-label",
        type=str,
        default=None,
        choices=["sa", "ga", "aco"],
        help="If set, only write rows with this final label (others are discarded).",
    )
    parser.add_argument(
        "--target-written",
        type=int,
        default=0,
        help="Stop once this many rows have been written (works for workers=1; for workers>1, requires --max-attempts and may overshoot slightly).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum number of attempts when using --target-written. If 0, uses --samples.",
    )
    args = parser.parse_args()

    config.ENABLE_VISUALIZATION = False

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    feature_names = DEFAULT_FEATURE_SPEC.names

    fieldnames: List[str] = (
        [f"f_{n}" for n in feature_names]
        + [
            "label",
            "sa_cost",
            "ga_cost",
            "aco_cost",
            "sa_sec",
            "ga_sec",
            "aco_sec",
            "budget",
            "seed",
        ]
    )

    skipped = 0
    written = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Parallel collection (process-based for CPU-bound optimizers).
        workers = max(1, int(args.workers))
        attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else int(args.samples)
        if workers == 1:
            i = 0
            while i < attempts and (int(args.target_written) <= 0 or written < int(args.target_written)):
                row = _collect_one(
                    i,
                    global_seed=int(args.seed),
                    budget=int(args.budget),
                    max_robots=int(args.max_robots),
                    min_path=int(args.min_path),
                    max_path=int(args.max_path),
                    quiet=bool(args.quiet),
                    label_policy=str(args.label_policy),
                    pareto_tol=float(args.pareto_tol),
                    time_weight=float(args.time_weight),
                    prefilter_edge_max=args.prefilter_edge_max,
                    keep_only_label=args.keep_only_label,
                    communication_min=float(args.communication_min),
                    communication_max=float(args.communication_max),
                    connectivity_min=float(args.connectivity_min),
                    connectivity_max=float(args.connectivity_max),
                )
                if row is None:
                    skipped += 1
                else:
                    writer.writerow(row)
                    written += 1
                    if args.log_every and (i + 1) % args.log_every == 0:
                        print(f"[{i+1}/{attempts}] wrote={written} skipped={skipped} last_label={row['label']}")
                i += 1
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=workers) as pool:
                worker_fn = functools.partial(
                    _collect_one,
                    global_seed=int(args.seed),
                    budget=int(args.budget),
                    max_robots=int(args.max_robots),
                    min_path=int(args.min_path),
                    max_path=int(args.max_path),
                    quiet=bool(args.quiet),
                    label_policy=str(args.label_policy),
                    pareto_tol=float(args.pareto_tol),
                    time_weight=float(args.time_weight),
                    prefilter_edge_max=args.prefilter_edge_max,
                    keep_only_label=args.keep_only_label,
                    communication_min=float(args.communication_min),
                    communication_max=float(args.communication_max),
                    connectivity_min=float(args.connectivity_min),
                    connectivity_max=float(args.connectivity_max),
                )
                iterator = pool.imap_unordered(
                    worker_fn,
                    range(attempts),
                    chunksize=max(1, int(args.chunksize)),
                )
                done = 0
                try:
                    for row in iterator:
                        done += 1
                        if row is None:
                            skipped += 1
                        else:
                            writer.writerow(row)
                            written += 1
                        if args.log_every and done % args.log_every == 0:
                            print(f"[{done}/{attempts}] wrote={written} skipped={skipped}")
                        if int(args.target_written) > 0 and written >= int(args.target_written):
                            pool.terminate()
                            break
                finally:
                    pool.join()

    print(f"Done. Output: {args.out} (wrote={written}, skipped={skipped})")


if __name__ == "__main__":
    main()
