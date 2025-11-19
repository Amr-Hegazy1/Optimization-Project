"""
Base class for optimization algorithms.

All optimization techniques should inherit from this class and implement
the required abstract methods to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
import math
import numpy as np
from config import *
from collections import deque


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
        
    def valid_move(self, p1, p2):
        """Check if p2 is the same cell or one of 4-connected adjacent cells."""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return dx + dy <= 1


    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def manhattan_distance(self, p1, p2):
        """Calculate Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def movements_to_positions(self, movements_array):
        """
        Convert movement indices to position paths.
       """
        # print(" Movements to positions input:", movements_array)
        path_array = []
        for r in range(R):
            x, y = ROBOT_INITIAL_POSITIONS[r]
            robot_path = []
            for move_idx in movements_array[r]:
                dx, dy = MOVES[move_idx]
                x, y = x + dx, y + dy
                # Ensure within map bounds
                x = max(0, min(MAP_HEIGHT - 1, x))
                y = max(0, min(MAP_WIDTH - 1, y))
                robot_path.append((x, y))
            path_array.append(robot_path)
        # print(" Movements to positions output:", path_array)
        return np.array(path_array, dtype=object)

    def compute_energy_used(self, path, initial_pos):
        """
        Compute cumulative energy (distance) used for a single robot path.
        Returns array of cumulative distances at each timestep.
        """
        energy = np.zeros(len(path))
        prev_pos = initial_pos

        for t, pos in enumerate(path):
            if t == 0:
                energy[t] = self.manhattan_distance(initial_pos, pos)
            else:
                energy[t] = energy[t - 1] + self.manhattan_distance(path[t - 1], pos)
            prev_pos = pos

        return energy


    def compute_link_weight(self, distance, R_c):
        """
        Compute distance-weighted link quality W_ij,t.
        W_ij,t = 1/(1+d_ij,t) if d <= R_c, else 0
        """
        if distance <= R_c:
            return 1.0 / (1.0 + distance)
        else:
            return 0.0


    def find_connected_components(self, adj_matrix):
        """
        Find connected components using BFS.
        Returns: list of sets, where each set contains robot indices in a component.
        """
        n_robots = len(adj_matrix)
        visited = set()
        components = []

        for start in range(n_robots):
            if start in visited:
                continue

            # BFS to find component
            component = set()
            queue = deque([start])
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


    def compute_disconnection_penalty(self, path_array):
        """
        Compute P_disconnect = Σ_t Σ_i δ_{i,t} * d_{i,net,t}
        where δ_{i,t} = 1 if robot i is disconnected from main network.
        """
        penalty = 0.0

        for t in range(PATH_LENGTH):
            # Get positions at time t
            positions_t = [path_array[i][t] for i in range(R)]
            
            # Build adjacency matrix based on connectivity threshold
            adj_matrix = np.zeros((R, R))
            for i in range(R):
                for j in range(i + 1, R):
                    dist = self.euclidean_distance(positions_t[i], positions_t[j])
                    if dist <= CONNECTIVITY_THRESHOLD:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1

            # Find connected components
            components = self.find_connected_components(adj_matrix)

            # Find main network (largest component)
            if len(components) == 1:
                # All connected, no penalty
                continue

            main_component = max(components, key=len)

            # For each robot not in main network, compute distance to main network
            for i in range(R):
                if i not in main_component:
                    # Robot i is disconnected
                    min_dist = float("inf")
                    for j in main_component:
                        dist = self.euclidean_distance(positions_t[i], positions_t[j])
                        min_dist = min(min_dist, dist)
                    penalty += min_dist

        return penalty


    def compute_obstacle_penalty(self, path_array):
        """
        Compute P_obstacle = Σ_t Σ_i η_{i,t}
        where η_{i,t} = 1 if robot i encounters obstacle at time t.
        """
        penalty = 0

        for r in range(R):
            for t in range(PATH_LENGTH):
                x, y = path_array[r][t]
                if MAP[x, y] == 2:  # Obstacle
                    penalty += 1

        return penalty


    def is_feasible(self, path_array):
        """
        Check if a path array satisfies all hard constraints.

        Constraints checked:
        1. Map bounds
        2. Motion constraint (4-connectivity, Manhattan distance <= 1)
        3. Energy budget
        4. Obstacle avoidance
        5. Collision avoidance (no two robots at same position at same time)

        Returns: True if feasible, False otherwise
        """

        # Track occupied positions at each timestep for collision detection
        occupied = {}

        for r in range(R):
            path = path_array[r]
            initial_pos = ROBOT_INITIAL_POSITIONS[r]

            # Check first step validity
            if not self.valid_move(initial_pos, path[0]):
                return False

            # Compute energy used
            energy_used = self.compute_energy_used(path, initial_pos)

            for t in range(PATH_LENGTH):
                x, y = path[t]

                # 1. Map bounds constraint
                if not (0 <= x < MAP_HEIGHT and 0 <= y < MAP_WIDTH):
                    return False

                # 2. Motion constraint (checked via valid_move)
                if t > 0:
                    if not self.valid_move(path[t - 1], path[t]):
                        return False

                # 3. Energy budget constraint
                if energy_used[t] > ENERGY_BUDGET:
                    return False

                # 4. Obstacle avoidance constraint
                if MAP[x, y] == 2:
                    return False

                # 5. Collision avoidance constraint
                if (x, y, t) in occupied:
                    return False
                occupied[(x, y, t)] = r

        return True


    def cost_function(self, path_array, visualize=False):
        """
        Compute the cost function value for a given path array.

        For minimization (SA standard):
        Cost = (α / Coverage) + (β / Connectivity) + (γ * P_disconnect)

        Note: Obstacles are handled by hard constraints in is_feasible()

        Returns: cost value (lower is better) if feasible, float('inf') if infeasible
        """
        
        # First check feasibility (includes obstacle avoidance)
        if not self.is_feasible(path_array):
            return float("inf")  # Infeasible solutions have infinite cost

        # 1. Coverage term: count of visited unexplored cells
        visited_unexplored = set()
        for r in range(R):
            for t in range(PATH_LENGTH):
                x, y = path_array[r][t]
                if MAP[x, y] == 0:  # Unexplored cell
                    visited_unexplored.add((x, y))

        coverage_count = len(visited_unexplored)

        # 2. Distance-weighted connectivity: Σ_t Σ_i Σ_j W_{ij,t}
        connectivity_sum = 0.0
        for t in range(PATH_LENGTH):
            positions_t = [path_array[i][t] for i in range(R)]
            for i in range(R):
                for j in range(i + 1, R):
                    dist = self.euclidean_distance(positions_t[i], positions_t[j])
                    connectivity_sum += self.compute_link_weight(dist, COMMUNICATION_RADIUS)

        # 3. Disconnection penalty: P_disconnect
        disconnection_penalty = self.compute_disconnection_penalty(path_array)

        # Compute cost: (α / Coverage) + (β / Connectivity) + (γ * P_disconnect)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6

        coverage_cost = ALPHA / (coverage_count + epsilon)
        connectivity_cost = BETA / (connectivity_sum + epsilon)
        disconnection_cost = GAMMA * disconnection_penalty

        cost = coverage_cost + connectivity_cost + disconnection_cost
        
        
        # Visualize if requested
        if visualize:
            self.visualize_coverage(visited_unexplored, path_array)
            print(f"\nCost Breakdown (lower is better):")
            print(f"  Coverage count: {coverage_count}")
            print(f"  Connectivity sum: {connectivity_sum:.2f}")
            print(f"  Disconnection penalty: {disconnection_penalty:.2f}")
            print(f"  ---")
            print(f"  Coverage cost (α/{coverage_count}): {coverage_cost:.6f}")
            print(
                f"  Connectivity cost (β/{connectivity_sum:.2f}): {connectivity_cost:.6f}"
            )
            print(
                f"  Disconnection cost (γ*{disconnection_penalty:.2f}): {disconnection_cost:.6f}"
            )
            print(f"  ---")
            print(f"  Total Cost: {cost:.6f}")

        return cost