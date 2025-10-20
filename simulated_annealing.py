import numpy as np
import random
from main import (
    movements_to_positions,
    cost_function,
    is_feasible,
    ROBOTS_POSITIONS,
    K,
)

class SimulatedAnnealing:
    def __init__(self, initial_temperature=100.0, cooling_rate=0.995, min_temperature=0.5, max_iterations=5000):
        self.initial_temperature = initial_temperature 
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations

    def generate_neighbor(self, movements_array, max_retries=20):
        """
        Generate a neighboring solution by randomly modifying movements.
        Tries up to max_retries to find a feasible neighbor.
        """
        for _ in range(max_retries):
            new_movements = [list(m) for m in movements_array]  # Copy current movements

            for r in range(len(new_movements)):
                # Randomly modify up to 10 movements for robot r
                for _ in range(10):
                    idx = random.randint(0, K - 1)
                    new_movements[r][idx] = random.randint(0, 4)

            # Convert movements back to path
            new_path = movements_to_positions(new_movements, ROBOTS_POSITIONS)

            # Check if the new path is feasible
            if is_feasible(new_path):
                return np.array(new_movements, dtype=object)

        # If no feasible path found, return the original path
        print("Warning: Could not find feasible solution  after max retries.")
        return movements_array

    def run(self, initial_movements):
        """Perform simulated annealing optimization."""
        current = initial_movements
        current_path = movements_to_positions(current, ROBOTS_POSITIONS)
        current_cost = cost_function(current_path)
        best_movements, best_cost = current, current_cost

        T = self.initial_temperature
        iteration = 0

        print(f"\n--- Simulated Annealing ---")
        print(f"Initial cost: {current_cost:.2f}")

        while T > self.min_temperature and iteration < self.max_iterations:
            new = self.generate_neighbor  (current)
            new_path = movements_to_positions(new, ROBOTS_POSITIONS)
            new_cost = cost_function(new_path)

            # Decide whether to accept the new solution or not
            if new_cost > current_cost or random.random() < np.exp((new_cost - current_cost) / T):
                current, current_cost = new, new_cost
                if new_cost > best_cost:
                    best_movements, best_cost = new, new_cost

            # Cool down the temperature using geometric cooling schedule
            T *= self.cooling_rate
            iteration += 1

            if iteration % 10 == 0:
                print(f"Iter {iteration:4d} | Temp: {T:6.3f} | Current: {current_cost:7.2f} | Best: {best_cost:7.2f}")

        print("\n--- Optimization Complete ---")
        print(f"Best cost found: {best_cost:.2f}")
        return best_movements, best_cost