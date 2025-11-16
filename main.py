import numpy as np
import math
import random
from collections import deque
import config
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


def compute_energy_used(path, initial_pos):
    """
    Compute cumulative energy (distance) used for a single robot path.
    Returns array of cumulative distances at each timestep.
    """
    energy = np.zeros(len(path))
    prev_pos = initial_pos

    for t, pos in enumerate(path):
        if t == 0:
            energy[t] = manhattan_distance(initial_pos, pos)
        else:
            energy[t] = energy[t - 1] + manhattan_distance(path[t - 1], pos)
        prev_pos = pos

    return energy


def compute_link_weight(distance, R_c):
    """
    Compute distance-weighted link quality W_ij,t.
    W_ij,t = 1/(1+d_ij,t) if d <= R_c, else 0
    """
    if distance <= R_c:
        return 1.0 / (1.0 + distance)
    else:
        return 0.0


def find_connected_components(adj_matrix):
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


def compute_disconnection_penalty(path_array):
    """
    Compute P_disconnect = Σ_t Σ_i δ_{i,t} * d_{i,net,t}
    where δ_{i,t} = 1 if robot i is disconnected from main network.
    """
    penalty = 0.0

    for t in range(K):
        # Get positions at time t
        positions_t = [path_array[i][t] for i in range(R)]

        # Build adjacency matrix based on connectivity threshold
        adj_matrix = np.zeros((R, R))
        for i in range(R):
            for j in range(i + 1, R):
                dist = euclidean_distance(positions_t[i], positions_t[j])
                if dist <= CONNECTIVITY_THRESHOLD:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1

        # Find connected components
        components = find_connected_components(adj_matrix)

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
                    dist = euclidean_distance(positions_t[i], positions_t[j])
                    min_dist = min(min_dist, dist)
                penalty += min_dist

    return penalty


def compute_obstacle_penalty(path_array):
    """
    Compute P_obstacle = Σ_t Σ_i η_{i,t}
    where η_{i,t} = 1 if robot i encounters obstacle at time t.
    """
    penalty = 0

    for r in range(R):
        for t in range(K):
            x, y = path_array[r][t]
            if MAP[x, y] == 2:  # Obstacle
                penalty += 1

    return penalty


def is_feasible(path_array):
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
        initial_pos = ROBOTS_POSITIONS[r]

        # Check first step validity
        if not valid_move(initial_pos, path[0]):
            return False

        # Compute energy used
        energy_used = compute_energy_used(path, initial_pos)

        for t in range(K):
            x, y = path[t]

            # 1. Map bounds constraint
            if not (0 <= x < N and 0 <= y < M):
                return False

            # 2. Motion constraint (checked via valid_move)
            if t > 0:
                if not valid_move(path[t - 1], path[t]):
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


def cost_function(path_array, visualize=False):
    """
    Compute the cost function value for a given path array.

    For minimization (SA standard):
    Cost = (α / Coverage) + (β / Connectivity) + (γ * P_disconnect)

    Note: Obstacles are handled by hard constraints in is_feasible()

    Returns: cost value (lower is better) if feasible, float('inf') if infeasible
    """

    # First check feasibility (includes obstacle avoidance)
    if not is_feasible(path_array):
        return float("inf")  # Infeasible solutions have infinite cost

    # 1. Coverage term: count of visited unexplored cells
    visited_unexplored = set()
    for r in range(R):
        for t in range(K):
            x, y = path_array[r][t]
            if MAP[x, y] == 0:  # Unexplored cell
                visited_unexplored.add((x, y))

    coverage_count = len(visited_unexplored)

    # 2. Distance-weighted connectivity: Σ_t Σ_i Σ_j W_{ij,t}
    connectivity_sum = 0.0
    for t in range(K):
        positions_t = [path_array[i][t] for i in range(R)]
        for i in range(R):
            for j in range(i + 1, R):
                dist = euclidean_distance(positions_t[i], positions_t[j])
                connectivity_sum += compute_link_weight(dist, COMMUNICATION_RADIUS)

    # 3. Disconnection penalty: P_disconnect
    disconnection_penalty = compute_disconnection_penalty(path_array)

    # Compute cost: (α / Coverage) + (β / Connectivity) + (γ * P_disconnect)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6

    coverage_cost = ALPHA / (coverage_count + epsilon)
    connectivity_cost = BETA / (connectivity_sum + epsilon)
    disconnection_cost = GAMMA * disconnection_penalty

    cost = coverage_cost + connectivity_cost + disconnection_cost

    # Visualize if requested
    if visualize:
        visualize_coverage(visited_unexplored, path_array)
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


def visualize_coverage(visited_unexplored, path_array):
    """Visualize the coverage map with visited cells and robot positions"""
    print("\nVisualizing Coverage Map...")

    # Get final positions
    final_pos = {}
    for r in range(R):
        final_position = tuple(path_array[r][-1])
        final_pos[final_position] = r + 1

    print("\n" + "=" * 50)
    print(f"Coverage Map ({N}x{M})")
    print("=" * 50)

    # Print column indices
    print("   ", end="")
    for j in range(M):
        print(f"{j:2}", end=" ")
    print()

    # Print grid with row indices
    for i in range(N):
        print(f"{i:2} ", end="")
        for j in range(M):
            if (i, j) in final_pos:
                print(f"R{final_pos[(i, j)]}", end=" ")
            elif (i, j) in visited_unexplored:
                print(" *", end=" ")
            elif MAP[i, j] == 2:
                print(" #", end=" ")
            else:
                print(" .", end=" ")
        print()

    # Print stats
    print("=" * 50)
    print(
        f"Cells explored: {len(visited_unexplored)}/{N * M} ({len(visited_unexplored) * 100 / (N * M):.1f}%)"
    )
    print(f"R* = Final robot position, * = Explored, # = Obstacle, . = Unexplored")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    # Import optimization algorithm and visualization
    # NOTE: SimulatedAnnealing inherits from optimization.base_optimizer.BaseOptimizer
    # NOTE: OptimizationVisualizer inherits from optimization.base_visualizer.BaseVisualizer
    from simulated_annealing import SimulatedAnnealing
    from genetic import GeneticOptimizer
    import threading

    print("\n" + "=" * 70)
    print("  MULTI-ROBOT PATH PLANNING")
    print("=" * 70)
    print("\nArchitecture:")
    print("  - BaseOptimizer: Abstract class for all optimization algorithms")
    print("  - BaseVisualizer: Abstract class for all visualizers")
    print("  - OptimizationVisualizer extends BaseVisualizer")
    print("=" * 70)

    # Create visualization window if enabled
    viz = None
    if config.ENABLE_VISUALIZATION:
        from genetic_visualization import GeneticVisualizer

        print("\nInitializing visualization...")
        viz = GeneticVisualizer(
            initial_positions=ROBOTS_POSITIONS,
            map_grid=MAP,
            communication_radius=COMMUNICATION_RADIUS,
            connectivity_threshold=CONNECTIVITY_THRESHOLD,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            visualization_step_size=config.VISUALIZATION_STEP_SIZE,
        )
    else:
        print("\nVisualization disabled - running optimization without GUI...")

    # Generate initial feasible path and convert it to movements
    print("Generating initial feasible solution...")
    initial_path = create_dummy_solution()
    initial_movements = positions_to_movements(initial_path, ROBOTS_POSITIONS)

    # Run optimization in a separate thread so GUI remains responsive
    def run_sa_optimization():
        # Run Simulated Annealing Optimization with visualization
        print(" - SIMULATED ANNEALING OPTIMIZATION started -")
        print("  - SimulatedAnnealing extends BaseOptimizer")
        sa = SimulatedAnnealing(
            initial_temperature=config.SA_INITIAL_TEMPERATURE,
            cooling_rate=config.SA_COOLING_RATE,
            min_temperature=config.SA_MIN_TEMPERATURE,
            max_iterations=config.SA_MAX_ITERATIONS,
            visualizer=viz,
        )

        # Enable fast mode if configured
        if (
            config.ENABLE_VISUALIZATION
            and hasattr(config, "FAST_MODE")
            and config.FAST_MODE
        ):
            sa.fast_mode = True
            print("\nFast Mode enabled - optimization will run at full speed")
            print("Visualization will replay after optimization completes\n")

        best_movements, best_cost = sa.run(initial_movements)

        # Convert best movements back to path
        best_path = movements_to_positions(best_movements, ROBOTS_POSITIONS)

        # Display final results in console
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        cost_function(best_path, visualize=True)
        print(f"\nFinal Best Cost (lower is better): {best_cost:.6f}")
        print("=" * 70)

        # If no visualization, exit after optimization
        if not config.ENABLE_VISUALIZATION:
            print("\nOptimization complete. Exiting...")

    # Run optimization in a separate thread so GUI remains responsive
    def run_ga_optimization():
        # Run Genetic Algorithm Optimization with visualization
        print(" - GENETIC ALGORITHM OPTIMIZATION started -")
        print("  - GeneticOptimizer extends BaseOptimizer")

        ga = GeneticOptimizer(
            population_size=20,
            generation_size=200,
            mutation_rate=0.3,
            elite_rate=0.1,
            visualizer=viz,
        )
        while NUMBER_OF_GENERATIONS > 0:
            # Enable fast mode if configured
            if (
                config.ENABLE_VISUALIZATION
                and hasattr(config, "FAST_MODE")
                and config.FAST_MODE
            ):
                ga.fast_mode = True
                print("\nFast Mode enabled - optimization will run at full speed")
                print("Visualization will replay after optimization completes\n")
            best_movements, best_cost = ga.run(
                initial_movements,
                mutation_method="swap_per_robot_path",
                parent_selection_method="sus",
                crossover_method="one_point_per_robots_paths",
                robot_positions=ROBOTS_POSITIONS,
            )

            # Convert best movements back to path
            best_path = movements_to_positions(best_movements, ROBOTS_POSITIONS)

        # Display final results in console
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        cost_function(best_path, visualize=True)
        print(f"\nFinal Best Cost (lower is better): {best_cost:.6f}")
        print("=" * 70)

        # If no visualization, exit after optimization
        if not config.ENABLE_VISUALIZATION:
            print("\nOptimization complete. Exiting...")

    # Start optimization
    print("Starting optimization...")
    if config.ENABLE_VISUALIZATION:
        print("Watch the real-time visualization window!\n")
        opt_thread = threading.Thread(target=run_ga_optimization, daemon=True)
        opt_thread.start()

        # Show visualization (this blocks until window is closed)
        viz.show()
    else:
        # Run directly without threading if no visualization
        run_ga_optimization()
