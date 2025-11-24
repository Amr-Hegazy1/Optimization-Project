"""
Case Study Comparison for Multi-Robot Path Planning Optimization Algorithms.

Simple comparison of SA, GA, and ACO algorithms with fixed common parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import case_study_config as cs_config
import os
from optimization.base_optimizer import BaseOptimizer
from ant_colony import AntColonyOptimizer
from genetic import GeneticOptimizer
from simulated_annealing import SimulatedAnnealing


# Global variables needed by base optimizer - will be set by case study
R = None
PATH_LENGTH = None
ALPHA = None
BETA = None
GAMMA = None
ENERGY_BUDGET = None
COMMUNICATION_RADIUS = None
CONNECTIVITY_THRESHOLD = None
MAP_HEIGHT = cs_config.MAP_HEIGHT
MAP_WIDTH = cs_config.MAP_WIDTH
MOVES = cs_config.MOVES
ROBOT_INITIAL_POSITIONS = cs_config.ROBOT_INITIAL_POSITIONS

# Map initialization
MAP = np.zeros((MAP_HEIGHT, MAP_WIDTH))


def run_sa(robot_positions, path_length, initial_temp, cooling_rate, min_temp, max_iterations):
    """Run Simulated Annealing."""
    print(f"\n{'='*70}")
    print("SIMULATED ANNEALING")
    print(f"{'='*70}")
    
    optimizer = SimulatedAnnealing(
        initial_temperature=initial_temp,
        cooling_rate=cooling_rate,
        min_temperature=min_temp,
        max_iterations=max_iterations,
        visualizer=None
    )
    
    # Generate initial solution
    initial_path = BaseOptimizer.create_dummy_solution(
        initial_positions=robot_positions,
        path_length=path_length
    )
    initial_movements = BaseOptimizer.positions_to_movements(initial_path, robot_positions)
    
    # Track cost history
    cost_history = []
    
    original_update = optimizer.update_visualization
    def tracking_update(iteration=0, best_cost=float('inf'), **kwargs):
        cost_history.append(best_cost)
        original_update(iteration=iteration, best_cost=best_cost, **kwargs)
    optimizer.update_visualization = tracking_update
    
    # Run
    best_movements, final_cost = optimizer.run(initial_movements)
    best_path = BaseOptimizer.movements_to_positions(best_movements, robot_positions)
    
    print(f"Final Cost: {final_cost:.6f}")
    
    return final_cost, cost_history, best_path


def run_ga(robot_positions, path_length, population_size, generation_size, mutation_rate, elite_rate):
    """Run Genetic Algorithm."""
    print(f"\n{'='*70}")
    print("GENETIC ALGORITHM")
    print(f"{'='*70}")
    
    optimizer = GeneticOptimizer(
        population_size=population_size,
        generation_size=generation_size,
        mutation_rate=mutation_rate,
        elite_rate=elite_rate,
        visualizer=None
    )
    
    # Generate initial solution
    initial_path = BaseOptimizer.create_dummy_solution(
        initial_positions=robot_positions,
        path_length=path_length
    )
    initial_movements = BaseOptimizer.positions_to_movements(initial_path, robot_positions)
    
    # Track cost history
    cost_history = []
    
    original_update = optimizer.update_visualization
    def tracking_update(iteration=0, best_cost=float('inf'), **kwargs):
        cost_history.append(best_cost)
        original_update(iteration=iteration, best_cost=best_cost, **kwargs)
    optimizer.update_visualization = tracking_update
    
    # Run
    best_movements, final_cost = optimizer.run(
        initial_movements,
        mutation_method='swap_per_robot_path',
        parent_selection_method='sus',
        crossover_method='one_point_per_robots_paths',
        robot_positions=robot_positions
    )
    best_path = BaseOptimizer.movements_to_positions(best_movements, robot_positions)
    
    print(f"Final Cost: {final_cost:.6f}")
    
    return final_cost, cost_history, best_path


def run_aco(robot_positions, path_length, num_ants, max_iterations, aco_alpha, aco_beta, evaporation_rate, strategy):
    """Run Ant Colony Optimization."""
    print(f"\n{'='*70}")
    print("ANT COLONY OPTIMIZATION")
    print(f"{'='*70}")
    
    optimizer = AntColonyOptimizer(
        num_ants=num_ants,
        max_iterations=max_iterations,
        alpha=aco_alpha,
        beta=aco_beta,
        evaporation_rate=evaporation_rate,
        strategy_name=strategy,
        visualizer=None
    )
    
    # Generate initial solution
    initial_path = BaseOptimizer.create_dummy_solution(
        initial_positions=robot_positions,
        path_length=path_length
    )
    initial_movements = BaseOptimizer.positions_to_movements(initial_path, robot_positions)
    
    # Track cost history
    cost_history = []
    
    original_update = optimizer.update_visualization
    def tracking_update(iteration=0, best_cost=float('inf'), **kwargs):
        cost_history.append(best_cost)
        original_update(iteration=iteration, best_cost=best_cost, **kwargs)
    optimizer.update_visualization = tracking_update
    
    # Run
    best_path = optimizer.run(initial_movements)
    final_cost = BaseOptimizer.cost_function(best_path)
    
    print(f"Final Cost: {final_cost:.6f}")
    
    return final_cost, cost_history, best_path


def plot_comparison(sa_history, ga_history, aco_history, num_robots, path_length, case_study_name):
    """Plot comparison of all three algorithms."""
    plt.figure(figsize=(14, 7))
    
    # Plot each algorithm with its own x-axis (iteration count)
    plt.plot(range(len(sa_history)), sa_history, label='Simulated Annealing', linewidth=2, marker='o', markersize=3, markevery=max(1, len(sa_history)//20))
    plt.plot(range(len(ga_history)), ga_history, label='Genetic Algorithm', linewidth=2, marker='s', markersize=3, markevery=max(1, len(ga_history)//20))
    plt.plot(range(len(aco_history)), aco_history, label='Ant Colony Optimization', linewidth=2, marker='^', markersize=3, markevery=max(1, len(aco_history)//20))
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Cost (lower is better)', fontsize=12)
    plt.title(f'Algorithm Comparison ({num_robots} robots, {path_length} steps)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Print iteration counts for debugging
    print(f"\nIteration counts:")
    print(f"  SA:  {len(sa_history)} iterations")
    print(f"  GA:  {len(ga_history)} iterations")
    print(f"  ACO: {len(aco_history)} iterations")
    
    # Ensure the directory exists
    os.makedirs("latex/Figures", exist_ok=True)
    
    # Save the plot
    filename = f"latex/Figures/{case_study_name}_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.close()  # Close the figure to free memory


def calculate_coverage(path_array, num_robots, path_length):
    """Calculate coverage percentage."""
    visited = set()
    for r in range(num_robots):
        for t in range(len(path_array[r])):  # Use actual path length, not the parameter
            x, y = path_array[r][t]
            if MAP[x, y] != 2:  # Not an obstacle
                visited.add((x, y))
    
    total_cells = MAP_HEIGHT * MAP_WIDTH
    coverage_pct = (len(visited) / total_cells) * 100
    return coverage_pct


def compare_all(case_study):
    """Compare all three algorithms using case study configuration."""
    
    # Setup - Update global variables from case study
    global R, PATH_LENGTH, ALPHA, BETA, GAMMA, ENERGY_BUDGET, COMMUNICATION_RADIUS, CONNECTIVITY_THRESHOLD
    
    R = case_study['num_robots']
    PATH_LENGTH = case_study['path_length']
    ALPHA = case_study['alpha']
    BETA = case_study['beta']
    GAMMA = case_study['gamma']
    ENERGY_BUDGET = case_study['energy_budget']
    COMMUNICATION_RADIUS = case_study['communication_radius']
    CONNECTIVITY_THRESHOLD = case_study['connectivity_threshold']
    
    robot_positions = cs_config.ROBOT_INITIAL_POSITIONS[:R]
    
    print(f"\n{'#'*70}")
    print(f"CASE STUDY: {case_study['name']}")
    print(f"Robots={R}, Steps={PATH_LENGTH}, Alpha={ALPHA}, Beta={BETA}")
    print(f"{'#'*70}")
    
    # Run SA
    print("\n[1/3] Running Simulated Annealing...")
    sa_cost, sa_history, sa_path = run_sa(
        robot_positions, PATH_LENGTH,
        case_study['sa_initial_temp'], 
        case_study['sa_cooling_rate'], 
        case_study['sa_min_temp'], 
        case_study['max_iterations']
    )
    sa_coverage = calculate_coverage(sa_path, R, PATH_LENGTH)
    print(f"SA completed: {len(sa_history)} iterations, Coverage: {sa_coverage:.2f}%")
    
    # Run GA
    print("\n[2/3] Running Genetic Algorithm...")
    ga_cost, ga_history, ga_path = run_ga(
        robot_positions, PATH_LENGTH,
        case_study['ga_population'], 
        case_study['max_iterations'], 
        case_study['ga_mutation_rate'], 
        case_study['ga_elite_rate']
    )
    ga_coverage = calculate_coverage(ga_path, R, PATH_LENGTH)
    print(f"GA completed: {len(ga_history)} iterations, Coverage: {ga_coverage:.2f}%")
    
    # Run ACO
    print("\n[3/3] Running Ant Colony Optimization...")
    aco_cost, aco_history, aco_path = run_aco(
        robot_positions, PATH_LENGTH,
        case_study['aco_num_ants'], 
        case_study['max_iterations'],
        case_study['aco_alpha'],
        case_study['aco_beta'],
        case_study['aco_evaporation'], 
        case_study['aco_strategy']
    )
    aco_coverage = calculate_coverage(aco_path, R, PATH_LENGTH)
    print(f"ACO completed: {len(aco_history)} iterations, Coverage: {aco_coverage:.2f}%")
    
    # Print comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Final Cost':<15} {'Coverage %':<15} {'Iterations':<15}")
    print(f"{'-'*70}")
    print(f"{'SA':<15} {sa_cost:<15.6f} {sa_coverage:<15.2f} {len(sa_history):<15}")
    print(f"{'GA':<15} {ga_cost:<15.6f} {ga_coverage:<15.2f} {len(ga_history):<15}")
    print(f"{'ACO':<15} {aco_cost:<15.6f} {aco_coverage:<15.2f} {len(aco_history):<15}")
    print(f"{'='*70}\n")
    
    # Plot comparison
    print("\nGenerating comparison plot...")
    plot_comparison(sa_history, ga_history, aco_history, R, PATH_LENGTH, case_study['name'])


def run_single_sa(num_robots, path_length, initial_temp, cooling_rate, min_temp, max_iterations):
    """Run only SA."""
    global R, PATH_LENGTH
    R = num_robots
    PATH_LENGTH = path_length
    
    robot_positions = cs_config.ROBOT_INITIAL_POSITIONS[:num_robots]
    print(f"\nRunning SA: {num_robots} robots, {path_length} steps")
    sa_cost, sa_history, sa_path = run_sa(robot_positions, path_length,
                                   initial_temp, cooling_rate, min_temp, max_iterations)
    sa_coverage = calculate_coverage(sa_path, num_robots, path_length)
    print(f"\nSA Result: Cost={sa_cost:.6f}, Coverage={sa_coverage:.2f}%")


def run_single_ga(num_robots, path_length, population_size, generation_size, mutation_rate, elite_rate):
    """Run only GA."""
    global R, PATH_LENGTH
    R = num_robots
    PATH_LENGTH = path_length
    
    robot_positions = cs_config.ROBOT_INITIAL_POSITIONS[:num_robots]
    print(f"\nRunning GA: {num_robots} robots, {path_length} steps")
    ga_cost, ga_history, ga_path = run_ga(robot_positions, path_length,
                                   population_size, generation_size, mutation_rate, elite_rate)
    ga_coverage = calculate_coverage(ga_path, num_robots, path_length)
    print(f"\nGA Result: Cost={ga_cost:.6f}, Coverage={ga_coverage:.2f}%")


def run_single_aco(num_robots, path_length, num_ants, max_iterations, aco_alpha, aco_beta, evaporation_rate, strategy):
    """Run only ACO."""
    global R, PATH_LENGTH
    R = num_robots
    PATH_LENGTH = path_length
    
    robot_positions = cs_config.ROBOT_INITIAL_POSITIONS[:num_robots]
    print(f"\nRunning ACO: {num_robots} robots, {path_length} steps")
    aco_cost, aco_history, aco_path = run_aco(robot_positions, path_length,
                                      num_ants, max_iterations, aco_alpha, aco_beta, evaporation_rate, strategy)
    aco_coverage = calculate_coverage(aco_path, num_robots, path_length)
    print(f"\nACO Result: Cost={aco_cost:.6f}, Coverage={aco_coverage:.2f}%")


# ============================================================================
# EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Run comparison for Case Study 1
    print("\n\n=== Running Case Study Comparisons ===\n")
    print("Starting Case Study 1 Comparison...")
    compare_all(cs_config.CASE_STUDY_1)
    
    # Run comparison for Case Study 2
    print("Starting Case Study 2 Comparison...")
    compare_all(cs_config.CASE_STUDY_2)
    
    # Run comparison for Case Study 3
    print("Starting Case Study 3 Comparison...")
    compare_all(cs_config.CASE_STUDY_3)
    
    # Run comparison for Case Study 4: Higher max_iterations
    print("Starting Case Study 4 Comparison...")
    compare_all(cs_config.CASE_STUDY_4)
    
    # Run comparison for Case Study 5: Higher SA initial temperature
    print("Starting Case Study 5 Comparison...")
    compare_all(cs_config.CASE_STUDY_5)
    
    # Run comparison for Case Study 6: Larger GA population
    print("Starting Case Study 6 Comparison...")
    compare_all(cs_config.CASE_STUDY_6)
    
    # Run comparison for Case Study 7: ACO with non-zero beta
    print("Starting Case Study 7 Comparison...")
    compare_all(cs_config.CASE_STUDY_7)
    
    # Run comparison for Case Study 8: Different global alpha and beta
    print("Starting Case Study 8 Comparison...")
    compare_all(cs_config.CASE_STUDY_8)
    
    # Run comparison for Case Study 9: Smaller communication radius
    print("Starting Case Study 9 Comparison...")
    compare_all(cs_config.CASE_STUDY_9)
    
    # Run comparison for Case Study 10: Fewer robots
    print("Starting Case Study 10 Comparison...")
    compare_all(cs_config.CASE_STUDY_10)
    
    # # Run comparison for Case Study 11: Longer path length
    # print("Starting Case Study 11 Comparison...")
    # compare_all(cs_config.CASE_STUDY_11)
