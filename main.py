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
from config import NUMBER_OF_GENERATIONS

# Import configuration parameters
N, M = config.MAP_HEIGHT, config.MAP_WIDTH
ROBOTS_POSITIONS = config.ROBOT_INITIAL_POSITIONS
positions = ROBOTS_POSITIONS
R = len(ROBOTS_POSITIONS)
K = config.PATH_LENGTH
V = config.ROBOT_VISION
NUMBER_OF_GENERATIONS = config.NUMBER_OF_GENERATIONS

# Energy and communication parameters
ENERGY_BUDGET = config.ENERGY_BUDGET
COMMUNICATION_RADIUS = config.COMMUNICATION_RADIUS
CONNECTIVITY_THRESHOLD = config.CONNECTIVITY_THRESHOLD

# Objective function weights
ALPHA = config.ALPHA
BETA = config.BETA
GAMMA = config.GAMMA
# Note: ZETA removed - obstacles handled by hard constraints

# Map initialization (0=unexplored, 1=free, 2=obstacle, 3=robot)

MOVES = config.MOVES

PHYSICAL_MAP = config.PHYSICAL_MAP


def init_map():
    """
    Initialize the map using Manhattan distance for vision range.
    Only cells within Manhattan distance V are visible.

    Returns:
        numpy.ndarray: Initialized map
    """
    map_grid = np.zeros((N, M), dtype=int)

    for robot_y, robot_x in ROBOTS_POSITIONS:
        for dy in range(-V, V + 1):
            for dx in range(-V, V + 1):
                # Check Manhattan distance
                if abs(dy) + abs(dx) <= V:
                    cell_y = robot_y + dy
                    cell_x = robot_x + dx

                    if 0 <= cell_y < N and 0 <= cell_x < M:
                        map_grid[cell_y, cell_x] = PHYSICAL_MAP[cell_y][cell_x]

    return map_grid


MAP = init_map()


def movements_to_positions(movements_array):
    """Convert from movement arrays [-> , <- ,..etc] to position paths [(x,y),..]."""
    global ROBOTS_POSITIONS
    path_array = []
    obstacle_found = False
    for k in range(K):
        current_positions = []
        for r in range(R):
            x, y = ROBOTS_POSITIONS[r]
            for move_idx in movements_array[r][k]:
                dx, dy = MOVES[move_idx]
                x, y = x + dx, y + dy
                # Ensure within map bounds
                x = max(0, min(N - 1, x))
                y = max(0, min(M - 1, y))

                if PHYSICAL_MAP[x][y] == 2:
                    obstacle_found = True
                    break

                current_positions.append((x, y))
            if not obstacle_found :
                ROBOTS_POSITIONS = current_positions
                path_array.append(current_positions)
    return np.array(path_array, dtype=object)


def positions_to_movements(path_array, initial_positions):
    """Convert from position paths [(x,y),..] to movement arrays [-> , <- ,..etc]."""
    movements_array = []
    for r, path in enumerate(path_array):
        x_prev, y_prev = initial_positions[r]
        robot_moves = []
        for x, y in path:
            dx, dy = x - x_prev, y - y_prev
            move_idx = (
                MOVES.index((dx, dy)) if (dx, dy) in MOVES else 4
            )  # stay in place if move is invalid
            robot_moves.append(move_idx)
            x_prev, y_prev = x, y
        movements_array.append(robot_moves)
    return np.array(movements_array, dtype=object)


def create_initial_random_path(max_attempts=1000):
    """
    Create a random, feasible path for each robot.
    Keeps trying until a path is found or max_attempts is reached.
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        path_array = []
        for start_x, start_y in ROBOTS_POSITIONS:
            robot_path = []
            x, y = start_x, start_y
            for _ in range(K):
                valid_moves = []
                # Generate list of valid moves within map bounds
                for move in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    nx, ny = x + move[0], y + move[1]
                    if 0 <= nx < N and 0 <= ny < M:
                        valid_moves.append(move)

                # Randomly select a valid move
                move = random.choice(valid_moves)
                x += move[0]
                y += move[1]
                robot_path.append((x, y))

            path_array.append(robot_path)

        path_array = np.array(path_array, dtype=object)

        # Check if the generated path is feasible
        if is_feasible(path_array):
            print(f"Found feasible initial path after {attempt} attempt(s).")
            return path_array

    # If no feasible path found after max_attempts
    print(
        "Warning: No feasible initial path found after many attempts. Returning last attempt."
    )
    return path_array


def create_dummy_solution():
    """
    Returns a dummy path for each robot.
    """
    path_array = create_initial_random_path()
    return np.array(path_array, dtype=object)


def valid_move(p1, p2):
    """Check if p2 is the same cell or one of 4-connected adjacent cells."""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return dx + dy <= 1


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1, p2):
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


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
