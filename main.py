import threading
from typing import Optional, Tuple

import numpy as np

import config
from optimization.base_optimizer import BaseOptimizer
from optimization.base_visualizer import BaseVisualizer
from ant_colony import AntColonyOptimizer
from genetic import GeneticOptimizer
from simulated_annealing import SimulatedAnnealing
from aco_visualization import AntColonyVisualizer
from genetic_visualization import GeneticVisualizer
from sa_visualization import OptimizationVisualizer


VISUALIZER_CLASSES = {
    "sa": OptimizationVisualizer,
    "ga": GeneticVisualizer,
    "aco": AntColonyVisualizer,
}

PathArray = np.ndarray


def build_visualizer(
    optimizer_key: str, map_grid: np.ndarray
) -> Optional[BaseVisualizer]:
    """Instantiate the appropriate visualizer if visualization is enabled."""
    if not config.ENABLE_VISUALIZATION:
        print("\nVisualization disabled - running optimization without GUI...")
        return None

    viz_cls = VISUALIZER_CLASSES.get(optimizer_key)
    if viz_cls is None:
        raise ValueError(f"Unsupported optimizer '{optimizer_key}' for visualization")

    print("\nInitializing visualization...")
    return viz_cls(
        initial_positions=config.ROBOT_INITIAL_POSITIONS,
        map_grid=map_grid,
        communication_radius=config.COMMUNICATION_RADIUS,
        connectivity_threshold=config.CONNECTIVITY_THRESHOLD,
        alpha=config.ALPHA,
        beta=config.BETA,
        gamma=config.GAMMA,
        visualization_step_size=config.VISUALIZATION_STEP_SIZE,
    )


def build_optimizer(
    optimizer_key: str, visualizer: Optional[BaseVisualizer]
) -> BaseOptimizer:
    """Create the optimizer instance configured from config.py."""
    if optimizer_key == "sa":
        return SimulatedAnnealing(
            initial_temperature=config.SA_INITIAL_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
            min_temperature=config.SA_MIN_TEMPERATURE,
            max_iterations=config.SA_MAX_ITERATIONS,
            visualizer=visualizer,
        )
    if optimizer_key == "ga":
        return GeneticOptimizer(
            population_size=config.GA_POPULATION_SIZE,
            generation_size=config.GA_GENERATION_SIZE,
            mutation_rate=config.GA_MUTATION_RATE,
            elite_rate=config.GA_ELITE_RATE,
            visualizer=visualizer,
        )
    if optimizer_key == "aco":
        return AntColonyOptimizer(
            num_ants=config.ACO_NUM_ANTS,
            max_iterations=config.ACO_MAX_ITERATIONS,
            alpha=config.ACO_ALPHA,
            beta=config.ACO_BETA,
            evaporation_rate=config.ACO_EVAPORATION_RATE,
            visualizer=visualizer,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_key}'")


def run_optimizer(
    optimizer_key: str, optimizer: BaseOptimizer, initial_movements: np.ndarray
) -> Tuple[PathArray, float]:
    """Execute the selected optimizer and return best path with its cost."""
    if optimizer_key == "sa":
        best_movements, best_cost = optimizer.run(initial_movements)
        best_path = BaseOptimizer.movements_to_positions(
            best_movements, config.ROBOT_INITIAL_POSITIONS
        )
        return best_path, best_cost

    if optimizer_key == "ga":
        best_movements, best_cost = optimizer.run(
            initial_movements,
            mutation_method=config.GA_MUTATION_METHOD,
            parent_selection_method=config.GA_PARENT_SELECTION_METHOD,
            crossover_method=config.GA_CROSSOVER_METHOD,
            robot_positions=config.ROBOT_INITIAL_POSITIONS,
        )
        best_path = BaseOptimizer.movements_to_positions(
            best_movements, config.ROBOT_INITIAL_POSITIONS
        )
        return best_path, best_cost

    if optimizer_key == "aco":
        best_path = optimizer.run(initial_movements)
        best_cost = BaseOptimizer.cost_function(best_path)
        return best_path, best_cost

    raise ValueError(f"Unsupported optimizer '{optimizer_key}'")


def summarize_results(best_path: PathArray, best_cost: float) -> None:
    """Print final optimization metrics and cost breakdown."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    BaseOptimizer.cost_function(best_path, visualize=True)
    print(f"\nFinal Best Cost (lower is better): {best_cost:.6f}")
    print("=" * 70)


def configure_fast_mode(optimizer: BaseOptimizer) -> None:
    """Enable fast mode on optimizers when visualization replay is desired."""
    if (
        config.ENABLE_VISUALIZATION
        and getattr(config, "FAST_MODE", False)
        and optimizer.visualizer is not None
    ):
        optimizer.fast_mode = True
        print("\nFast Mode enabled - optimization will run at full speed")
        print("Visualization will replay after optimization completes\n")


def main() -> None:
    optimizer_key = config.OPTIMIZER_TYPE.lower()

    print("\n" + "=" * 70)
    print("  MULTI-ROBOT PATH PLANNING")
    print("=" * 70)
    print("\nArchitecture:")
    print("  - BaseOptimizer: Abstract class for all optimization algorithms")
    print("  - BaseVisualizer: Abstract class for all visualizers")
    print("  - Concrete visualizers extend BaseVisualizer")
    print("=" * 70)

    visualizer = build_visualizer(optimizer_key, config.MAP)

    print("Generating initial feasible solution...")
    initial_path = BaseOptimizer.create_dummy_solution()
    initial_movements = BaseOptimizer.positions_to_movements(
        initial_path, config.ROBOT_INITIAL_POSITIONS
    )

    optimizer = build_optimizer(optimizer_key, visualizer)
    configure_fast_mode(optimizer)

    def optimization_task() -> None:
        best_path, best_cost = run_optimizer(
            optimizer_key, optimizer, initial_movements
        )
        summarize_results(best_path, best_cost)
        if not config.ENABLE_VISUALIZATION:
            print("\nOptimization complete. Exiting...")

    print("Starting optimization...")
    if config.ENABLE_VISUALIZATION and visualizer is not None:
        print("Watch the real-time visualization window!\n")
        opt_thread = threading.Thread(target=optimization_task, daemon=True)
        opt_thread.start()
        visualizer.show()
    else:
        optimization_task()


if __name__ == "__main__":
    main()
