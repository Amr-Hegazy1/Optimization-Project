import numpy as np
import random
from typing import Tuple

import config
from optimization.base_optimizer import BaseOptimizer


class SimulatedAnnealing(BaseOptimizer):
    """
    Simulated Annealing optimization algorithm for multi-robot path planning.
    
    This algorithm uses a probabilistic approach to escape local optima by
    accepting worse solutions with a temperature-dependent probability.
    """
    
    def __init__(self, initial_temperature=10.0, cooling_rate=0.995, 
                min_temperature=0.1, max_iterations=5000, visualizer=None):
        """
        Initialize Simulated Annealing optimizer.
        
        Args:
            initial_temperature (float): Starting temperature for annealing
            cooling_rate (float): Rate at which temperature decreases (0 < rate < 1)
            min_temperature (float): Minimum temperature threshold for stopping
            max_iterations (int): Maximum number of iterations
            visualizer: Optional visualization object
        """
        super().__init__(max_iterations=max_iterations, visualizer=visualizer)
        self.initial_temperature = initial_temperature 
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.current_temperature = initial_temperature

    def generate_neighbor(
        self, movements_array: np.ndarray, max_retries: int = 100
    ) -> np.ndarray:
        """
        Generate a neighboring solution by randomly modifying movements.
        
        Implementation of abstract method from BaseOptimizer.
        Tries up to max_retries to find a feasible neighbor.
        
        Args:
            movements_array: Current movement array solution
            max_retries (int): Maximum attempts to find feasible neighbor
            
        Returns:
            np.ndarray: Neighboring movement array
        """
        for _ in range(max_retries):
            new_movements = [list(m) for m in movements_array]  # Copy current movements

            for r in range(len(new_movements)):
                # Randomly modify up to 10 movements for robot r
                for _ in range(15):
                    idx = random.randint(0, config.PATH_LENGTH - 1)
                    new_movements[r][idx] = random.randint(0, 4)

            # Convert movements back to path
            new_path = self.movements_to_positions(
                new_movements, config.ROBOT_INITIAL_POSITIONS
            )

            # Check if the new path is feasible
            if self.is_feasible(
                new_path, initial_positions=config.ROBOT_INITIAL_POSITIONS
            ):
                return np.array(new_movements, dtype=object)

        # If no feasible path found, return the original path
        print("Warning: Could not find feasible solution  after max retries.")
        return movements_array

    def acceptance_criterion(self, current_cost: float, new_cost: float) -> bool:
        """
        Determine whether to accept a new solution using Metropolis criterion.
        
        Implementation of abstract method from BaseOptimizer.
        For MINIMIZATION: Accepts improvements always (new_cost < current_cost),
        and accepts worse solutions with probability exp(-(new_cost - current_cost) / T).
        
        Args:
            current_cost (float): Cost of current solution
            new_cost (float): Cost of candidate solution
            
        Returns:
            bool: True if new solution should be accepted
        """
        if new_cost < current_cost:  # Improvement (lower cost is better)
            return True
        else:
            # Probabilistic acceptance for worse solutions
            acceptance_probability = np.exp(-(new_cost - current_cost) / self.current_temperature)
            return random.random() < acceptance_probability

    def run(self, initial_movements: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Perform simulated annealing optimization.
        
        Implementation of abstract method from BaseOptimizer.
        Minimizes cost function (lower cost is better).
        
        Args:
            initial_movements: Initial movement array solution
            
        Returns:
            tuple: (best_movements, best_cost)
        """
        current = initial_movements
        current_path = self.movements_to_positions(
            current, config.ROBOT_INITIAL_POSITIONS
        )
        current_cost = self.cost_function(current_path)
        best_movements, best_cost = current, current_cost
        self.best_solution = best_movements
        self.best_cost = best_cost

        T = self.initial_temperature
        self.current_temperature = T
        iteration = 0
        self.current_iteration = 0

        print(f"\n--- Simulated Annealing (Minimization) ---")
        print(f"Initial cost: {current_cost:.6f}")

        while T > self.min_temperature and iteration < self.max_iterations:
            new = self.generate_neighbor(current)
            new_path = self.movements_to_positions(
                new, config.ROBOT_INITIAL_POSITIONS
            )
            new_cost = self.cost_function(new_path)

            # Use the acceptance criterion method
            if self.acceptance_criterion(current_cost, new_cost):
                current, current_cost = new, new_cost
                if new_cost < best_cost:  # Lower cost is better (minimization)
                    best_movements, best_cost = new, new_cost
                    self.best_solution = best_movements
                    self.best_cost = best_cost

            # Update visualization using base class method
            self.update_visualization(
                iteration=iteration,
                temperature=T,
                current_cost=current_cost,
                best_cost=best_cost,
                best_path=self.movements_to_positions(
                    best_movements, config.ROBOT_INITIAL_POSITIONS
                ),
                current_path=self.movements_to_positions(
                    current, config.ROBOT_INITIAL_POSITIONS
                )
            )
            
            # Wait for visualization using base class method
            self.wait_for_visualization()

            # Cool down the temperature using geometric cooling schedule
            T *= self.cooling_rate
            
            # Cool down the temperature using linear cooling schedule
            #delta_T = (self.initial_temperature - self.min_temperature) / self.max_iterations
            #T = max(self.min_temperature, T - delta_T)
            
            self.current_temperature = T
            iteration += 1
            self.current_iteration = iteration

            if iteration % 10 == 0:
                print(f"Iter {iteration:4d} | Temp: {T:6.3f} | Current: {current_cost:7.6f} | Best: {best_cost:7.6f}")

        # Notify visualization that optimization is complete using base class method
        self.finish_optimization()

        print("\n--- Optimization Complete ---")
        print(f"Best cost found: {best_cost:.6f}")
        return best_movements, best_cost
    
    def get_hyperparameters(self):
        """
        Get SA-specific hyperparameters.
        
        Overrides base class method to include SA-specific parameters.
        
        Returns:
            dict: Dictionary of hyperparameter names and values
        """
        params = super().get_hyperparameters()
        params.update({
            'initial_temperature': self.initial_temperature,
            'cooling_rate': self.cooling_rate,
            'min_temperature': self.min_temperature,
            'current_temperature': self.current_temperature
        })
        return params