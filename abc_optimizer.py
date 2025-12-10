"""
Artificial Bee Colony optimizer adapted for discrete multi-robot paths.

Implements employed/onlooker/scout phases with feasibility-aware neighbors.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

import config
from optimization.base_optimizer import BaseOptimizer

MovementArray = np.ndarray
PathArray = np.ndarray


class ABCOptimizer(BaseOptimizer):
    """Artificial Bee Colony optimizer using discrete neighbor perturbations."""

    def __init__(
        self,
        colony_size: int = config.ABC_COLONY_SIZE,
        max_cycles: int = config.ABC_MAX_CYCLES,
        limit: int = config.ABC_LIMIT,
        onlooker_ratio: float = config.ABC_ONLOOKER_RATIO,
        neighbor_window: int = config.ABC_NEIGHBOR_WINDOW,
        neighbor_attempts: int = config.ABC_NEIGHBOR_ATTEMPTS,
        visualizer=None,
    ) -> None:
        super().__init__(max_iterations=max_cycles, visualizer=visualizer)
        self.colony_size = colony_size
        self.limit = limit
        self.onlooker_ratio = onlooker_ratio
        self.neighbor_window = neighbor_window
        self.neighbor_attempts = neighbor_attempts

    def _evaluate(self, movements: MovementArray) -> Tuple[PathArray, float]:
        """Return path and cost for a movement matrix."""
        path = self.movements_to_positions(movements, config.ROBOT_INITIAL_POSITIONS)
        cost = self.cost_function(path)
        return path, cost

    def _random_feasible_movements(self) -> MovementArray:
        """Generate a random feasible movement matrix."""
        path = self.create_dummy_solution(
            initial_positions=config.ROBOT_INITIAL_POSITIONS,
            path_length=config.PATH_LENGTH,
        )
        return self.positions_to_movements(path, config.ROBOT_INITIAL_POSITIONS)

    def _generate_neighbor(
        self,
        source: MovementArray,
        peer: MovementArray,
    ) -> MovementArray:
        """Create a neighbor by perturbing a time window and optional swaps."""
        for _ in range(self.neighbor_attempts):
            candidate = [list(r) for r in source]
            robot_idx = random.randrange(len(candidate))
            path_len = len(candidate[robot_idx])
            window = min(self.neighbor_window, path_len)
            start = random.randrange(path_len)
            end = min(path_len, start + window)

            # Blend peer moves and random moves inside the window.
            for t in range(start, end):
                if peer is not None and random.random() < 0.6 and t < len(peer[robot_idx]):
                    candidate[robot_idx][t] = peer[robot_idx][t]
                else:
                    candidate[robot_idx][t] = random.randrange(len(config.MOVES))

            # Occasional swap on another robot to diversify.
            if random.random() < 0.3:
                other_robot = random.randrange(len(candidate))
                idx1 = random.randrange(len(candidate[other_robot]))
                idx2 = random.randrange(len(candidate[other_robot]))
                candidate[other_robot][idx1], candidate[other_robot][idx2] = (
                    candidate[other_robot][idx2],
                    candidate[other_robot][idx1],
                )

            candidate_movements = np.array(candidate, dtype=object)
            candidate_path = self.movements_to_positions(candidate_movements, config.ROBOT_INITIAL_POSITIONS)
            if self.is_feasible(candidate_path, initial_positions=config.ROBOT_INITIAL_POSITIONS, path_length=config.PATH_LENGTH):
                return candidate_movements
        return source

    def _fitness(self, cost: float) -> float:
        if not np.isfinite(cost):
            return 0.0
        return 1.0 / (cost + 1e-6)

    def run(self, initial_movements: Optional[MovementArray] = None) -> Tuple[MovementArray, float]:
        """Execute the ABC optimization loop."""
        # Initialize colony
        sources: List[MovementArray] = []
        trials: List[int] = []
        paths: List[PathArray] = []
        costs: List[float] = []

        if initial_movements is None:
            initial_movements = self._random_feasible_movements()

        sources.append(initial_movements)
        path, cost = self._evaluate(initial_movements)
        paths.append(path)
        costs.append(cost)
        trials.append(0)

        while len(sources) < self.colony_size:
            m = self._random_feasible_movements()
            p, c = self._evaluate(m)
            sources.append(m)
            paths.append(p)
            costs.append(c)
            trials.append(0)

        best_cost = float("inf")
        best_movements = sources[0]
        best_path = paths[0]

        for cycle in range(self.max_iterations):
            # Employed bees
            for i in range(self.colony_size):
                peer_idx = random.randrange(self.colony_size)
                while peer_idx == i:
                    peer_idx = random.randrange(self.colony_size)
                neighbor = self._generate_neighbor(sources[i], sources[peer_idx])
                neighbor_path, neighbor_cost = self._evaluate(neighbor)
                if neighbor_cost < costs[i]:
                    sources[i] = neighbor
                    paths[i] = neighbor_path
                    costs[i] = neighbor_cost
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker bees
            fitnesses = [self._fitness(c) for c in costs]
            total_fitness = sum(fitnesses)
            num_onlookers = max(1, int(self.onlooker_ratio * self.colony_size))

            if total_fitness > 0:
                for _ in range(num_onlookers):
                    r = random.uniform(0, total_fitness)
                    cumulative = 0.0
                    selected_idx = 0
                    for idx, fit in enumerate(fitnesses):
                        cumulative += fit
                        if cumulative >= r:
                            selected_idx = idx
                            break

                    peer_idx = random.randrange(self.colony_size)
                    while peer_idx == selected_idx:
                        peer_idx = random.randrange(self.colony_size)

                    neighbor = self._generate_neighbor(sources[selected_idx], sources[peer_idx])
                    neighbor_path, neighbor_cost = self._evaluate(neighbor)
                    if neighbor_cost < costs[selected_idx]:
                        sources[selected_idx] = neighbor
                        paths[selected_idx] = neighbor_path
                        costs[selected_idx] = neighbor_cost
                        trials[selected_idx] = 0
                    else:
                        trials[selected_idx] += 1

            # Scout phase
            for i in range(self.colony_size):
                if trials[i] >= self.limit:
                    sources[i] = self._random_feasible_movements()
                    paths[i], costs[i] = self._evaluate(sources[i])
                    trials[i] = 0

            # Track best
            iteration_best_idx = int(np.argmin(costs))
            iteration_best_cost = costs[iteration_best_idx]
            iteration_best_path = paths[iteration_best_idx]

            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_movements = sources[iteration_best_idx]
                best_path = iteration_best_path
                self.best_solution = best_movements
                self.best_cost = best_cost

            # Visualization/update hooks
            self.update_visualization(
                iteration=cycle,
                current_cost=iteration_best_cost,
                best_cost=best_cost,
                best_path=best_path,
                current_path=iteration_best_path,
            )
            self.wait_for_visualization()

            if (cycle + 1) % 10 == 0:
                print(
                    f"Cycle {cycle + 1:4d} | Iter best: {iteration_best_cost:8.5f} | Global best: {best_cost:8.5f}"
                )

        self.finish_optimization()
        return best_movements, best_cost

    # Abstract method implementations (not used directly in ABC loop).
    def generate_neighbor(self, solution):
        """Fallback neighbor: perturb solution using itself as peer."""
        return self._generate_neighbor(solution, solution)

    def acceptance_criterion(self, current_cost, new_cost):
        """Greedy acceptance helper."""
        return new_cost < current_cost

    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params.update(
            {
                "colony_size": self.colony_size,
                "limit": self.limit,
                "onlooker_ratio": self.onlooker_ratio,
                "neighbor_window": self.neighbor_window,
                "neighbor_attempts": self.neighbor_attempts,
            }
        )
        return params
