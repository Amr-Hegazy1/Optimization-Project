"""
Base class for optimization algorithms.

All optimization techniques should inherit from this class and implement
the required abstract methods to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
import math
import random
from collections import deque
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import config
from config import *

Position = Tuple[int, int]
MovementSequence = Sequence[int]
MovementArray = Sequence[MovementSequence]
PathArray = np.ndarray


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    This class defines the interface that all optimization algorithms must follow,
    ensuring consistency across different optimization techniques (SA, GA, PSO, etc.).
    
    Attributes:
        max_iterations (int): Maximum number of optimization iterations
        visualizer: Optional visualization object for real-time updates
    """
    
    def __init__(self, max_iterations=5000, visualizer=None):
        """
        Initialize the base optimizer.
        
        Args:
            max_iterations (int): Maximum number of iterations
            visualizer: Optional visualizer object that implements update methods
        """
        self.max_iterations = max_iterations
        self.visualizer = visualizer
        self.current_iteration = 0
        self.best_solution = None
        self.best_cost = np.inf  # Initialize to infinity for minimization
        
        # History tracking for fast mode replay
        self.optimization_history = []
        self.fast_mode = False
        
    @abstractmethod
    def run(self, initial_solution):
        """
        Run the optimization algorithm.
        
        Args:
            initial_solution: Initial solution to start optimization from
            
        Returns:
            tuple: (best_solution, best_cost)
        """
        pass
    
    @abstractmethod
    def generate_neighbor(self, solution):
        """
        Generate a neighboring solution.
        
        This method should implement the neighborhood structure specific to
        the optimization algorithm.
        
        Args:
            solution: Current solution
            
        Returns:
            A neighboring solution
        """
        pass
    
    @abstractmethod
    def acceptance_criterion(self, current_cost, new_cost):
        """
        Determine whether to accept a new solution.
        
        Different algorithms have different acceptance criteria:
        - SA: probabilistic based on temperature
        - Hill Climbing: only accept improvements
        - GA: selection based on fitness
        
        Args:
            current_cost (float): Cost of current solution
            new_cost (float): Cost of candidate solution
            
        Returns:
            bool: True if new solution should be accepted
        """
        pass
    
    def update_visualization(self, **kwargs):
        """
        Update the visualization with current optimization state.
        
        This method should be called during optimization to provide real-time
        feedback. Subclasses can override to provide additional parameters.
        
        Args:
            **kwargs: Algorithm-specific visualization parameters
        """
        if self.fast_mode:
            # In fast mode, store history instead of updating visualization
            self.optimization_history.append(kwargs.copy())
        elif self.visualizer is not None:
            self.visualizer.update_optimization(**kwargs)
    
    def wait_for_visualization(self):
        """
        Wait for visualization to complete current animation.
        
        This ensures synchronization between optimization and visualization.
        """
        if self.fast_mode:
            # In fast mode, don't wait for visualization
            return
        if self.visualizer is not None and hasattr(self.visualizer, 'wait_for_animation_complete'):
            self.visualizer.wait_for_animation_complete()
    
    def finish_optimization(self):
        """
        Signal that optimization is complete.
        
        This allows the visualizer to update its state and display final results.
        """
        if self.fast_mode and self.visualizer is not None:
            # In fast mode, replay the optimization history
            if hasattr(self.visualizer, 'replay_optimization_history'):
                self.visualizer.replay_optimization_history(self.optimization_history)
        elif self.visualizer is not None and hasattr(self.visualizer, 'finish_optimization'):
            self.visualizer.finish_optimization()
    
    def get_algorithm_name(self):
        """
        Get the name of the optimization algorithm.
        
        Returns:
            str: Name of the algorithm (e.g., "Simulated Annealing", "Genetic Algorithm")
        """
        return self.__class__.__name__
    
    def get_hyperparameters(self):
        """
        Get algorithm-specific hyperparameters as a dictionary.
        
        Subclasses should override this to return their specific parameters.
        
        Returns:
            dict: Dictionary of hyperparameter names and values
        """
        return {
            'max_iterations': self.max_iterations
        }
        
    @staticmethod
    def valid_move(p1: Position, p2: Position) -> bool:
        """Return True when p2 is identical to or 4-connected with p1."""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return dx + dy <= 1

    @staticmethod
    def euclidean_distance(p1: Position, p2: Position) -> float:
        """Compute Euclidean distance between two grid positions."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def manhattan_distance(p1: Position, p2: Position) -> int:
        """Compute Manhattan distance between two grid positions."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def movements_to_positions(
        movements_array: MovementArray,
        initial_positions: Optional[Sequence[Position]] = None,
    ) -> PathArray:
        """Convert movement indices to concrete position paths."""
        initial_positions = initial_positions or ROBOT_INITIAL_POSITIONS
        path_array: List[List[Position]] = []
        for r, moves in enumerate(movements_array):
            x, y = initial_positions[r]
            robot_path: List[Position] = []
            for move_idx in moves:
                dx, dy = MOVES[move_idx]
                x = max(0, min(MAP_HEIGHT - 1, x + dx))
                y = max(0, min(MAP_WIDTH - 1, y + dy))
                robot_path.append((x, y))
            path_array.append(robot_path)
        return np.array(path_array, dtype=object)

    @staticmethod
    def positions_to_movements(
        path_array: Sequence[Sequence[Position]],
        initial_positions: Optional[Sequence[Position]] = None,
    ) -> np.ndarray:
        """Convert absolute paths back to movement indices."""
        initial_positions = initial_positions or ROBOT_INITIAL_POSITIONS
        movements_array: List[List[int]] = []
        for r, path in enumerate(path_array):
            x_prev, y_prev = initial_positions[r]
            robot_moves: List[int] = []
            for x, y in path:
                dx, dy = x - x_prev, y - y_prev
                move_idx = MOVES.index((dx, dy)) if (dx, dy) in MOVES else len(MOVES) - 1
                robot_moves.append(move_idx)
                x_prev, y_prev = x, y
            movements_array.append(robot_moves)
        return np.array(movements_array, dtype=object)

    @classmethod
    def create_initial_random_path(
        cls,
        max_attempts: int = 1000,
        initial_positions: Optional[Sequence[Position]] = None,
        path_length: Optional[int] = None,
    ) -> PathArray:
        """Generate a random feasible path array for all robots."""
        initial_positions = initial_positions or ROBOT_INITIAL_POSITIONS
        path_length = path_length or PATH_LENGTH
        attempt = 0
        candidate: PathArray = np.empty((0,), dtype=object)
        while attempt < max_attempts:
            attempt += 1
            path_array: List[List[Position]] = []
            for start_x, start_y in initial_positions:
                robot_path: List[Position] = []
                x, y = start_x, start_y
                for _ in range(path_length):
                    valid_moves = []
                    for move in MOVES:
                        nx, ny = x + move[0], y + move[1]
                        if 0 <= nx < MAP_HEIGHT and 0 <= ny < MAP_WIDTH and MAP[nx, ny] != 2:
                            valid_moves.append(move)
                    move_dx, move_dy = random.choice(valid_moves)
                    x += move_dx
                    y += move_dy
                    robot_path.append((x, y))
                path_array.append(robot_path)
            candidate = np.array(path_array, dtype=object)
            if cls.is_feasible(candidate, initial_positions=initial_positions, path_length=path_length):
                print(f"Found feasible initial path after {attempt} attempt(s).")
                return candidate
        print("Warning: No feasible initial path found after many attempts. Returning last attempt.")
        return candidate

    @classmethod
    def create_dummy_solution(
        cls,
        max_attempts: int = 1000,
        initial_positions: Optional[Sequence[Position]] = None,
        path_length: Optional[int] = None,
    ) -> PathArray:
        """Return a feasible path array suitable for bootstrapping optimizers."""
        return cls.create_initial_random_path(
            max_attempts=max_attempts,
            initial_positions=initial_positions,
            path_length=path_length,
        )

    @staticmethod
    def compute_energy_used(path: Sequence[Position], initial_pos: Position) -> np.ndarray:
        """Compute cumulative Manhattan distance per timestep for a single path."""
        energy = np.zeros(len(path))
        for t, pos in enumerate(path):
            if t == 0:
                energy[t] = BaseOptimizer.manhattan_distance(initial_pos, pos)
            else:
                energy[t] = energy[t - 1] + BaseOptimizer.manhattan_distance(path[t - 1], pos)
        return energy

    @staticmethod
    def compute_link_weight(distance: float, communication_radius: float) -> float:
        """Return inverse-distance link quality within the communication radius."""
        if distance <= communication_radius:
            return 1.0 / (1.0 + distance)
        return 0.0

    @staticmethod
    def find_connected_components(adj_matrix: np.ndarray) -> List[Set[int]]:
        """Return connected components for an adjacency matrix using BFS."""
        n_robots = len(adj_matrix)
        visited: Set[int] = set()
        components: List[Set[int]] = []
        for start in range(n_robots):
            if start in visited:
                continue
            component: Set[int] = set()
            queue: deque[int] = deque([start])
            component.add(start)
            visited.add(start)
            while queue:
                node = queue.popleft()
                for neighbor in range(n_robots):
                    if adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return components

    @classmethod
    def compute_disconnection_penalty(
        cls,
        path_array: PathArray,
        path_length: Optional[int] = None,
    ) -> float:
        """Return cumulative distance robots must travel to reconnect to the main network."""
        penalty = 0.0
        horizon = path_array.shape[1]
        for t in range(horizon):
            positions_t = [path_array[i][t] for i in range(path_array.shape[0])]
            adj_matrix = np.zeros((path_array.shape[0], path_array.shape[0]))
            for i in range(path_array.shape[0]):
                for j in range(i + 1, path_array.shape[0]):
                    dist = cls.euclidean_distance(positions_t[i], positions_t[j])
                    if dist <= CONNECTIVITY_THRESHOLD:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
            components = cls.find_connected_components(adj_matrix)
            if len(components) == 1:
                continue
            main_component = max(components, key=len)
            for i in range(path_array.shape[0]):
                if i in main_component:
                    continue
                min_dist = min(
                    cls.euclidean_distance(positions_t[i], positions_t[j]) for j in main_component
                )
                penalty += min_dist
        return penalty

    @classmethod
    def compute_obstacle_penalty(
        cls, path_array: PathArray, path_length: Optional[int] = None
    ) -> int:
        """Count visits to obstacle cells across all robots and timesteps."""
        penalty = 0
        horizon = path_length or PATH_LENGTH
        for r in range(path_array.shape[0]):
            for t in range(horizon):
                x, y = path_array[r][t]
                if MAP[x, y] == 2:
                    penalty += 1
        return penalty

    @classmethod
    def is_feasible(
        cls,
        path_array: PathArray,
        initial_positions: Optional[Sequence[Position]] = None,
        path_length: Optional[int] = None,
    ) -> bool:
        """Validate all hard constraints (bounds, motion, energy, obstacles, collisions)."""
        initial_positions = initial_positions or ROBOT_INITIAL_POSITIONS
        horizon = path_array.shape[1]
        occupied = {}

        for r in range(path_array.shape[0]):
            path = path_array[r]
            initial_pos = initial_positions[r]
            if not cls.valid_move(initial_pos, path[0]):
                return False
            energy_used = cls.compute_energy_used(path, initial_pos)
            for t in range(horizon):
                x, y = path[t]
                if not (0 <= x < MAP_HEIGHT and 0 <= y < MAP_WIDTH):
                    return False
                if t > 0 and not cls.valid_move(path[t - 1], path[t]):
                    return False
                if energy_used[t] > ENERGY_BUDGET:
                    return False
                if MAP[x, y] == 2:
                    return False
                if (x, y, t) in occupied:
                    return False
                occupied[(x, y, t)] = r
        return True

    @classmethod
    def cost_function(cls, path_array: PathArray, visualize: bool = False) -> float:
        """Evaluate the multi-objective cost for a path array; lower is better."""
        if not cls.is_feasible(path_array):
            return float("inf")
        visited_unexplored = set()
        for r in range(path_array.shape[0]):
            for t in range(path_array.shape[1]):
                x, y = path_array[r][t]
                if INITIAL_MAP[x, y] == 0:
                    visited_unexplored.add((x, y))
        coverage_count = len(visited_unexplored)
        connectivity_sum = 0.0
        for t in range(path_array.shape[1]):
            positions_t = [path_array[i][t] for i in range(path_array.shape[0])]
            for i in range(path_array.shape[0]):
                for j in range(i + 1, path_array.shape[0]):
                    dist = cls.euclidean_distance(positions_t[i], positions_t[j])
                    connectivity_sum += cls.compute_link_weight(dist, COMMUNICATION_RADIUS)
        disconnection_penalty = cls.compute_disconnection_penalty(path_array)
        epsilon = 1e-6
        coverage_cost = ALPHA / (coverage_count + epsilon)
        connectivity_cost = BETA / (connectivity_sum + epsilon)
        disconnection_cost = GAMMA * disconnection_penalty
        cost = coverage_cost + connectivity_cost + disconnection_cost
        if visualize:
            cls.visualize_coverage(visited_unexplored, path_array)
            print("\nCost Breakdown (lower is better):")
            print(f"  Coverage count: {coverage_count}")
            print(f"  Connectivity sum: {connectivity_sum:.2f}")
            print(f"  Disconnection penalty: {disconnection_penalty:.2f}")
            print("  ---")
            print(f"  Coverage cost (α/{coverage_count}): {coverage_cost:.6f}")
            print(
                f"  Connectivity cost (β/{connectivity_sum:.2f}): {connectivity_cost:.6f}"
            )
            print(
                f"  Disconnection cost (γ*{disconnection_penalty:.2f}): {disconnection_cost:.6f}"
            )
            print("  ---")
            print(f"  Total Cost: {cost:.6f}")
        return cost

    @staticmethod
    def visualize_coverage(visited_unexplored: Set[Position], path_array: PathArray) -> None:
        """Print a textual coverage summary including final robot positions."""
        print("\nVisualizing Coverage Map...")
        final_positions = {}
        for r in range(path_array.shape[0]):
            final_pos = tuple(path_array[r][-1])
            final_positions[final_pos] = r + 1
        print("\n" + "=" * 50)
        print(f"Coverage Map ({MAP_HEIGHT}x{MAP_WIDTH})")
        print("=" * 50)
        print("   ", end="")
        for j in range(MAP_WIDTH):
            print(f"{j:2}", end=" ")
        print()
        for i in range(MAP_HEIGHT):
            print(f"{i:2} ", end="")
            for j in range(MAP_WIDTH):
                if (i, j) in final_positions:
                    print(f"R{final_positions[(i, j)]}", end=" ")
                elif (i, j) in visited_unexplored:
                    print(" *", end=" ")
                elif MAP[i, j] == 2:
                    print(" #", end=" ")
                else:
                    print(" .", end=" ")
            print()
        print("=" * 50)
        total_cells = MAP_HEIGHT * MAP_WIDTH
        explored_pct = len(visited_unexplored) * 100 / total_cells
        print(f"Cells explored: {len(visited_unexplored)}/{total_cells} ({explored_pct:.1f}%)")
        print("R* = Final robot position, * = Explored, # = Obstacle, . = Unexplored")
        print("=" * 50 + "\n")

    @staticmethod
    def simulate_movements(movements_array, initial_positions=None):
        """
        Simulate robot movements step by step, updating maps along the way.

        Args:
            movements_array: Array of movement sequences for each robot
            initial_positions: Starting positions for robots (defaults to ROBOT_INITIAL_POSITIONS)

        Returns:
            tuple: (final_positions, path_array) - Final robot positions and full path history
        """
        initial_positions = initial_positions or config.ROBOT_INITIAL_POSITIONS
        robots_positions = list(initial_positions)
        path_array = []

        for k in range(config.PATH_LENGTH):
            old_positions = robots_positions.copy()
            current_positions = []
            obstacle_found = False

            for r in range(R):
                x, y = robots_positions[r]
                move_idx = movements_array[r][k]
                dx, dy = MOVES[move_idx]
                x, y = x + dx, y + dy

                # Ensure within map bounds
                x = max(0, min(config.MAP_HEIGHT - 1, x))
                y = max(0, min(config.MAP_WIDTH - 1, y))

                if PHYSICAL_MAP[x][y] == 2:
                    obstacle_found = True
                    break

                current_positions.append((x, y))

            if not obstacle_found:
                # Update both maps with old and new positions
                BaseOptimizer.update_map(old_positions, current_positions)
                robots_positions = current_positions
                path_array.append(current_positions)
            else:
                # If obstacle found, stop simulation
                break

        return robots_positions, np.array(path_array, dtype=object)

    @staticmethod
    def update_map(old_positions: Sequence[Position], new_positions: Sequence[Position]) -> None:
        """
        Update the physical map and virtual map after robots move.

        This function:
        1. Removes robots from their old positions in PHYSICAL_MAP (sets to free cells)
        2. Places robots at their new positions in PHYSICAL_MAP
        3. Updates the virtual MAP to reflect new robot vision ranges

        Args:
            old_positions: Previous robot positions [(x,y), ...]
            new_positions: New robot positions [(x,y), ...]
        """
        # Remove old robot positions from PHYSICAL_MAP (set to free cells)
        for old_x, old_y in old_positions:
            if 0 <= old_x < MAP_HEIGHT and 0 <= old_y < MAP_WIDTH:
                if PHYSICAL_MAP[old_x][old_y] == CELL_ROBOT:
                    PHYSICAL_MAP[old_x][old_y] = CELL_FREE

        # Place robots at new positions in PHYSICAL_MAP
        for new_x, new_y in new_positions:
            if 0 <= new_x < MAP_HEIGHT and 0 <= new_y < MAP_WIDTH:
                PHYSICAL_MAP[new_x][new_y] = CELL_ROBOT

        # Update virtual MAP with new vision ranges
        BaseOptimizer.update_virtual_map(new_positions)

    @staticmethod
    def update_virtual_map(robot_positions: Sequence[Position]) -> None:
        """
        Update the virtual MAP to reflect robot vision from new positions.

        This function updates the MAP (virtual map) based on what robots can see
        from their current positions using Manhattan distance vision range.

        Args:
            robot_positions: Current robot positions [(x,y), ...]
        """
        # For each robot, update the virtual map with what it can see
        for robot_y, robot_x in robot_positions:
            # Check all cells within vision range using Manhattan distance
            for dy in range(-ROBOT_VISION, ROBOT_VISION + 1):
                for dx in range(-ROBOT_VISION, ROBOT_VISION + 1):
                    # Check if within Manhattan distance vision range
                    if abs(dy) + abs(dx) <= ROBOT_VISION:
                        cell_y = robot_y + dy
                        cell_x = robot_x + dx

                        # Check if cell is within map bounds
                        if 0 <= cell_y < MAP_HEIGHT and 0 <= cell_x < MAP_WIDTH:
                            # Update virtual MAP with what's in PHYSICAL_MAP
                            # Only update if not already known (or update to latest)
                            MAP[cell_y, cell_x] = PHYSICAL_MAP[cell_y][cell_x]

