"""Ant Colony Optimization with pluggable strategy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Set, Tuple

import numpy as np

from config import *
from optimization.base_optimizer import BaseOptimizer
from optimization.base_visualizer import BaseVisualizer

Position = Tuple[int, int]
FeasibleMove = Tuple[int, Position]
PathArray = np.ndarray


class AntColonyStrategy(ABC):
    """Shared utilities for the different ACO variants."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        heuristic_map: np.ndarray | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = 1.0
        if heuristic_map is None:
            heuristic_map = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=float)
        self.heuristic_map = heuristic_map

    @staticmethod
    def _sample_move(weights: np.ndarray, feasible_moves: Sequence[FeasibleMove]) -> int:
        """Classic roulette-wheel sampling."""
        if not feasible_moves:
            raise ValueError("No feasible moves provided to strategy.")
        if len(feasible_moves) == 1:
            return feasible_moves[0][0]
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            random_index = int(np.random.randint(len(feasible_moves)))
            return feasible_moves[random_index][0]
        probabilities = weights / total_weight
        cumulative = np.cumsum(probabilities)
        r = float(np.random.rand())
        chosen = int(np.searchsorted(cumulative, r, side="right"))
        chosen = min(chosen, len(feasible_moves) - 1)
        return feasible_moves[chosen][0]

    def before_iteration(self, pheromone_map: np.ndarray) -> None:
        """Default behavior: apply uniform evaporation before constructing solutions."""
        pheromone_map *= (1.0 - self.evaporation_rate)

    def local_after_move(
        self,
        pheromone_map: np.ndarray,
        _current_pos: Position,
        _next_pos: Position,
    ) -> None:
        """Hook that runs immediately after every move (optional)."""
        return

    def update_local_pheromone(
        self,
        pheromone_map: np.ndarray,
        ant_path: PathArray,
        path_cost: float,
    ) -> None:
        """Hook for depositing pheromone after an ant constructs a full path."""
        return

    def after_iteration(
        self,
        pheromone_map: np.ndarray,
        iteration_best_path: PathArray,
        iteration_best_cost: float,
    ) -> None:
        """Hook for global updates once all ants finish an iteration."""
        return

    @abstractmethod
    def select_move(
        self,
        pheromone_map: np.ndarray,
        current_pos: Position,
        feasible_moves: Sequence[FeasibleMove],
        visited_positions: Set[Position],
    ) -> int:
        """Return the movement index to take next for the current robot."""


class SACOStrategy(AntColonyStrategy):
    """Simple Ant Colony strategy that relies purely on pheromones."""

    def __init__(self, alpha: float, evaporation_rate: float) -> None:
        super().__init__(alpha=alpha, beta=0.0, evaporation_rate=evaporation_rate)

    def select_move(
        self,
        pheromone_map: np.ndarray,
        _current_pos: Position,
        feasible_moves: Sequence[FeasibleMove],
        _visited_positions: Set[Position],
    ) -> int:
        weights = np.array(
            [max(1e-9, pheromone_map[x, y] ** self.alpha) for _, (x, y) in feasible_moves],
            dtype=float,
        )
        return self._sample_move(weights, feasible_moves)

    def update_local_pheromone(
        self,
        pheromone_map: np.ndarray,
        ant_path: PathArray,
        path_cost: float,
    ) -> None:
        if not np.isfinite(path_cost):
            return
        contribution = 1.0 / (1.0 + path_cost)
        visited: Set[Position] = set()
        for robot_path in ant_path:
            for x, y in robot_path:
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                pheromone_map[x, y] += contribution


class ASStrategy(AntColonyStrategy):
    """Ant System strategy that blends pheromones with heuristic desirability."""

    def __init__(
        self,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        heuristic_map: np.ndarray,
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, evaporation_rate=evaporation_rate, heuristic_map=heuristic_map)
        self.local_decay = min(max(evaporation_rate, 0.05), 0.9)

    def select_move(
        self,
        pheromone_map: np.ndarray,
        _current_pos: Position,
        feasible_moves: Sequence[FeasibleMove],
        visited_positions: Set[Position],
    ) -> int:
        candidate_moves = [move for move in feasible_moves if move[1] not in visited_positions]
        if not candidate_moves:
            candidate_moves = list(feasible_moves)
        weights = np.array(
            [
                max(1e-9, pheromone_map[x, y] ** self.alpha)
                * max(1e-9, self.heuristic_map[x, y] ** self.beta)
                for _, (x, y) in candidate_moves
            ],
            dtype=float,
        )
        return self._sample_move(weights, candidate_moves)

    def local_after_move(
        self,
        pheromone_map: np.ndarray,
        _current_pos: Position,
        next_pos: Position,
    ) -> None:
        x, y = next_pos
        pheromone_map[x, y] = (1.0 - self.local_decay) * pheromone_map[x, y] + self.local_decay * self.initial_pheromone

    def update_local_pheromone(
        self,
        pheromone_map: np.ndarray,
        ant_path: PathArray,
        path_cost: float,
    ) -> None:
        if not np.isfinite(path_cost):
            return
        contribution = 1.0 / (1.0 + path_cost)
        for robot_path in ant_path:
            for x, y in robot_path:
                pheromone_map[x, y] += contribution * self.heuristic_map[x, y]

    def after_iteration(
        self,
        pheromone_map: np.ndarray,
        iteration_best_path: PathArray,
        iteration_best_cost: float,
    ) -> None:
        if not np.isfinite(iteration_best_cost):
            return
        reinforcement = (1.0 + self.beta) / (1.0 + iteration_best_cost)
        for robot_path in iteration_best_path:
            for x, y in robot_path:
                pheromone_map[x, y] = (1.0 - self.evaporation_rate) * pheromone_map[x, y] + self.evaporation_rate * reinforcement


class AntColonyOptimizer(BaseOptimizer):
    """Ant Colony optimizer that delegates behavior to interchangeable strategies."""

    def __init__(
        self,
        max_iterations: int = ACO_MAX_ITERATIONS,
        visualizer: BaseVisualizer | None = None,
        alpha: float = ACO_ALPHA,
        beta: float = ACO_BETA,
        evaporation_rate: float = ACO_EVAPORATION_RATE,
        num_ants: int = ACO_NUM_ANTS,
        strategy_name: str = ACO_STRATEGY,
    ) -> None:
        super().__init__(max_iterations, visualizer)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.num_ants = num_ants
        self.ants = np.zeros((self.num_ants, R, PATH_LENGTH), dtype=int)
        self.pheromone_map = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=float)
        heuristic_map = self._build_heuristic_map()
        self.strategy = self._build_strategy(strategy_name, heuristic_map)

    def _build_heuristic_map(self) -> np.ndarray:
        """Heuristic preference: unexplored > free > obstacles."""
        heuristic = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=float)
        heuristic[MAP == CELL_UNEXPLORED] = 2.0
        heuristic[MAP == CELL_FREE] = 1.2
        heuristic[MAP == CELL_OBSTACLE] = 1e-6
        return heuristic

    def _build_strategy(self, strategy_name: str, heuristic_map: np.ndarray) -> AntColonyStrategy:
        """Return the configured strategy instance."""
        normalized_name = strategy_name.lower().strip()
        if normalized_name == "as":
            beta_value = self.beta if self.beta > 0 else 1.0
            return ASStrategy(self.alpha, beta_value, self.evaporation_rate, heuristic_map)
        return SACOStrategy(self.alpha, self.evaporation_rate)

    def _get_feasible_moves(self, position: Position) -> List[FeasibleMove]:
        """Return movement indices that keep the robot within bounds."""
        feasible: List[FeasibleMove] = []
        for move_index, (dx, dy) in enumerate(MOVES):
            nx, ny = position[0] + dx, position[1] + dy
            if 0 <= nx < MAP_HEIGHT and 0 <= ny < MAP_WIDTH:
                feasible.append((move_index, (nx, ny)))
        return feasible

    def _construct_ant_solution(self, ant_index: int) -> None:
        """Sample moves for every robot for a single ant."""
        visited_per_robot: List[Set[Position]] = [set() for _ in range(R)]
        for robot_idx in range(R):
            current_pos = ROBOT_INITIAL_POSITIONS[robot_idx]
            visited_per_robot[robot_idx].add(current_pos)
            for step in range(PATH_LENGTH):
                feasible_moves = self._get_feasible_moves(current_pos)
                # Select the next move using the current strategy (SACO or AS).
                move_index = self.strategy.select_move(
                    self.pheromone_map,
                    current_pos,
                    feasible_moves,
                    visited_per_robot[robot_idx],
                )
                self.ants[ant_index, robot_idx, step] = move_index
                dx, dy = MOVES[move_index]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                # Some strategies evaporate/deposit pheromone per move.
                self.strategy.local_after_move(self.pheromone_map, current_pos, next_pos)
                current_pos = next_pos
                visited_per_robot[robot_idx].add(current_pos)

    def _evaluate_ant(self, ant_index: int) -> Tuple[PathArray, float]:
        """Build a full solution for one ant and return its path/cost."""
        self._construct_ant_solution(ant_index)
        path = self.movements_to_positions(self.ants[ant_index])
        cost = self.cost_function(path)
        self.strategy.update_local_pheromone(self.pheromone_map, path, cost)
        return path, cost

    def run(self, initial_solution=None):
        """Execute the Ant Colony optimization process."""
        best_cost = float("inf")
        best_solution = None
        best_path = None

        for iteration in range(self.max_iterations):
            self.strategy.before_iteration(self.pheromone_map)
            ant_paths: List[PathArray] = []
            ant_costs: List[float] = []

            for ant_index in range(self.num_ants):
                path, cost = self._evaluate_ant(ant_index)
                ant_paths.append(path)
                ant_costs.append(cost)

            costs = np.array(ant_costs, dtype=float)
            best_idx = int(np.argmin(costs))
            current_cost = float(costs[best_idx])
            current_path = ant_paths[best_idx]

            self.strategy.after_iteration(self.pheromone_map, current_path, current_cost)

            if current_cost <= best_cost:
                best_cost = current_cost
                best_solution = self.ants[best_idx].copy()
                best_path = current_path

            if self.visualizer is not None:
                update = {
                    "iteration": iteration,
                    "current_cost": current_cost,
                    "best_cost": best_cost,
                    "best_path": best_path,
                    "avg_pheromone": float(np.mean(self.pheromone_map)),
                    "max_pheromone": float(np.max(self.pheromone_map)),
                    "current_path": current_path,
                }
                if self.fast_mode:
                    self.optimization_history.append(update)
                else:
                    self.visualizer.update_optimization(**update)
                    self.wait_for_visualization()

        self.finish_optimization()
        if best_path is not None:
            return best_path
        if best_solution is not None:
            return self.movements_to_positions(best_solution)
        return None

    def generate_neighbor(self, solution):
        return super().generate_neighbor(solution)

    def acceptance_criterion(self, current_cost, new_cost):
        return super().acceptance_criterion(current_cost, new_cost)
