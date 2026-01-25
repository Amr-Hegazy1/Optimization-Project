import sys
import os
import time
import numpy as np
from typing import List, Tuple
import argparse
import random

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from simulated_annealing import SimulatedAnnealing
from genetic import GeneticOptimizer
from ant_colony import AntColonyOptimizer
from abc_optimizer import ABCOptimizer

def run_benchmark(algorithm_name: str, num_runs: int = 20):
    print(f"\nStarting benchmark for {algorithm_name} with {num_runs} runs...")
    
    costs = []
    times = []
    solutions = []
    
    # Configuration for Case C1
    # SA: Alpha=3000, Beta=500, Temp=10.0, Iterations=910 (from Table 4.1)
    # GA: Alpha=3000, Beta=500, Pop=20, Gen=200 (from Table 4.4)
    # ACO: Alpha=0.5, Beta=1.0, Evap=0.4, Ants=30, Iter=200 (from Table 4.7 Case A1 - assuming this is the best ACO)
    
    # Override config for Case C1
    config.PATH_LENGTH = 125
    config.ALPHA = 3000
    config.BETA = 500
    
    # Ensure we have 6 robots
    # config.ROBOT_INITIAL_POSITIONS is already 6 robots in config.py
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}", end='\r')
        
        start_time = time.time()
        
        if algorithm_name == "SA":
            optimizer = SimulatedAnnealing(
                initial_temperature=10.0,
                max_iterations=910, # From Case C1 in SA Results
                visualizer=None
            )
            # SA needs an initial solution
            initial_movements = np.random.randint(0, 5, size=(config.R, config.PATH_LENGTH))
            best_movements, best_cost = optimizer.run(initial_movements)
            best_solution = optimizer.movements_to_positions(best_movements, config.ROBOT_INITIAL_POSITIONS)
            
        elif algorithm_name == "GA":
            optimizer = GeneticOptimizer(
                population_size=20, # From Case C1 in GA Results
                generation_size=200, # From Case C1 in GA Results
                mutation_rate=0.1, # From Case C10 (best mut/elite) or default? Table 4.4 says C1 is best. 
                                   # Table 4.6 says C10 is best for mut/elite variation. 
                                   # Text says "Case C1 ... low mutation (10%) and elitism (10%)".
                elite_rate=0.1,
                visualizer=None
            )
            # GA generates its own population
            initial_movements = np.random.randint(0, 5, size=(config.R, config.PATH_LENGTH))
            best_movements, best_cost = optimizer.run(initial_movements)
            best_solution = optimizer.movements_to_positions(best_movements, config.ROBOT_INITIAL_POSITIONS)
            
        elif algorithm_name == "ACO":
            optimizer = AntColonyOptimizer(
                max_iterations=200, # From Case A1
                num_ants=30,        # From Case A1
                alpha=0.5,          # From Case A1
                beta=1.0,           # From Case A1
                evaporation_rate=0.4, # From Case A1
                strategy_name="as",
                visualizer=None
            )
            # ACO constructs solutions
            best_path = optimizer.run()
            # ACO run returns path, we need cost. 
            # But wait, optimizer.run() in ACO returns best_path (positions), not movements.
            # And we need cost.
            if best_path is not None:
                best_cost = optimizer.cost_function(best_path)
                best_solution = best_path # Store path as solution for ACO
            else:
                best_cost = float('inf')
                best_solution = None
        
        elif algorithm_name == "ABC":
            optimizer = ABCOptimizer(
                colony_size=24,
                max_cycles=200,
                limit=12,
                onlooker_ratio=0.5,
                neighbor_window=10,
                neighbor_attempts=5,
                visualizer=None,
            )
            best_movements, best_cost = optimizer.run(initial_movements=None)
            best_solution = optimizer.movements_to_positions(best_movements, config.ROBOT_INITIAL_POSITIONS)

        end_time = time.time()
        
        costs.append(best_cost)
        times.append(end_time - start_time)
        solutions.append(best_solution)
        
    print(f"Run {num_runs}/{num_runs} completed.")
    
    return {
        "costs": costs,
        "times": times,
        "solutions": solutions
    }

def format_solution(solution):
    # Solution is now expected to be a path array of shape (R, Steps, 2)
    if solution is None:
        return "N/A"
    
    try:
        # Check if it's a numpy array
        if isinstance(solution, np.ndarray):
            # Assuming shape (R, Steps, 2) or (R, Steps+1, 2)
            # We want the final position for each robot
            final_positions = []
            for r in range(len(solution)):
                # Get the last position
                last_pos = solution[r][-1]
                final_positions.append(tuple(last_pos))
            
            # Format as string: (x1, y1), (x2, y2), ...
            # Truncate if too long
            formatted = ", ".join([str(p) for p in final_positions])
            if len(formatted) > 20:
                return formatted[:17] + "..."
            return formatted
    except Exception as e:
        return "Error"

    return "..."

def main():
    parser = argparse.ArgumentParser(description='Run optimization benchmarks.')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs per algorithm')
    parser.add_argument('--test', action='store_true', help='Run in test mode (reduced parameters)')
    parser.add_argument(
        '--algos',
        nargs='+',
        default=["SA", "GA", "ACO", "ABC"],
        choices=["SA", "GA", "ACO", "ABC"],
        help='Algorithms to benchmark (subset of: SA GA ACO ABC)',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    num_runs = args.runs
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    if args.test:
        print("Running in TEST mode with reduced parameters.")
        config.PATH_LENGTH = 20
        # Override other config if needed for speed
        
    results = {}
    
    name_map = {
        "SA": "Simulated Annealing",
        "GA": "Genetic Algorithm",
        "ACO": "Ant Colony Optimization",
        "ABC": "Artificial Bee Colony",
    }

    for algo in args.algos:
        results[name_map[algo]] = run_benchmark(algo, num_runs)
    
    # Print Table
    print("\n" + "="*80)
    header = f"{'Metric':<25} | " + " | ".join([f"{name_map[a]:<20}" for a in args.algos])
    print(header)
    print("-" * (len(header) + 5))
    
    metrics = [
        "Optimal Solution",
        "Optimal Fitness Value",
        "Mean Fitness",
        "Standard Deviation",
        "Computational Time"
    ]
    
    # Calculate metrics
    data = {}
    for algo in results:
        costs = results[algo]["costs"]
        times = results[algo]["times"]
        solutions = results[algo]["solutions"]
        
        best_idx = np.argmin(costs)
        optimal_fitness = costs[best_idx]
        mean_fitness = np.mean(costs)
        std_dev = np.std(costs)
        avg_time = np.mean(times)
        best_solution_overall = solutions[best_idx]
        
        data[algo] = {
            "Optimal Solution": format_solution(best_solution_overall),
            "Optimal Fitness Value": f"{optimal_fitness:.4f}",
            "Mean Fitness": f"{mean_fitness:.4f}",
            "Standard Deviation": f"{std_dev:.4e}",
            "Computational Time": f"{avg_time:.3f} sec"
        }

    for metric in metrics:
        row = f"{metric:<25} | "
        for algo in [name_map[a] for a in args.algos]:
            val = data[algo].get(metric, "N/A")
            row += f"{val:<20} | "
        print(row)
    print("="*80)

if __name__ == "__main__":
    main()
