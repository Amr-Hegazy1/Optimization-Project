import numpy as np
import math
from collections import deque

# Map dimensions
N, M = 10, 20

# Robot configuration
ROBOTS_POSITIONS = [(0, 0), (0, 1)]
R = len(ROBOTS_POSITIONS)
K = 5  # Number of steps in the generated path

# Energy and communication parameters
ENERGY_BUDGET = 50  # ℓ: maximum cells each robot can traverse
COMMUNICATION_RADIUS = 10.0  # R_c: communication radius threshold
CONNECTIVITY_THRESHOLD = 10.0  # d_threshold: for determining if edge exists

# Objective function weights
ALPHA = 1.0  # Coverage weight
BETA = 0.5   # Distance-weighted connectivity weight
GAMMA = 2.0  # Disconnection penalty weight
ZETA = 5.0   # Obstacle encounter penalty weight

# Map initialization (0=unexplored, 1=free, 2=obstacle, 3=robot)
Map = np.zeros((N, M))


def create_dummy_solution():
    """
    Returns a dummy path for each robot.
    """
    path_array = [[(1,0),(2,0),(3,0),(4,0),(5,0)],
                  [(0,1),(0,2),(0,3),(0,4),(0,5)]]
    return np.array(path_array, dtype=object)


def valid_move(p1, p2):
    """Check if p2 is the same cell or one of 4-connected adjacent cells."""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return (dx + dy <= 1)


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
            energy[t] = energy[t-1] + manhattan_distance(path[t-1], pos)
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
                min_dist = float('inf')
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
            if Map[x, y] == 2:  # Obstacle
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
                if not valid_move(path[t-1], path[t]):
                    return False
            
            # 3. Energy budget constraint
            if energy_used[t] > ENERGY_BUDGET:
                return False
            
            # 4. Obstacle avoidance constraint
            if Map[x, y] == 2:
                return False
            
            # 5. Collision avoidance constraint
            if (x, y, t) in occupied:
                return False
            occupied[(x, y, t)] = r
    
    return True


def cost_function(path_array, visualize=False):
    """
    Compute the objective function value for a given path array.
    
    Objective: f = α * Coverage + β * Connectivity - γ * P_disconnect - ζ * P_obstacle
    
    Returns: objective value if feasible, -1 if infeasible
    """
    
    # First check feasibility
    if not is_feasible(path_array):
        return -1
    
    # 1. Coverage term: α * Σ C_{x,y}
    visited_unexplored = set()
    for r in range(R):
        for t in range(K):
            x, y = path_array[r][t]
            if Map[x, y] == 0:  # Unexplored cell
                visited_unexplored.add((x, y))
    
    coverage_term = ALPHA * len(visited_unexplored)
    
    # 2. Distance-weighted connectivity term: β * Σ_t Σ_i Σ_j W_{ij,t}
    connectivity_term = 0.0
    for t in range(K):
        positions_t = [path_array[i][t] for i in range(R)]
        for i in range(R):
            for j in range(i + 1, R):
                dist = euclidean_distance(positions_t[i], positions_t[j])
                connectivity_term += compute_link_weight(dist, COMMUNICATION_RADIUS)
    
    connectivity_term *= BETA
    
    # 3. Disconnection penalty: γ * P_disconnect
    disconnection_penalty = GAMMA * compute_disconnection_penalty(path_array)
    
    # 4. Obstacle encounter penalty: ζ * P_obstacle
    obstacle_penalty = ZETA * compute_obstacle_penalty(path_array)
    
    # Compute total objective
    objective = (coverage_term + connectivity_term - 
                 disconnection_penalty - obstacle_penalty)
    
    # Visualize if requested
    if visualize:
        visualize_coverage(visited_unexplored, path_array)
        print(f"\nObjective Breakdown:")
        print(f"  Coverage term (α={ALPHA}): {coverage_term:.2f}")
        print(f"  Connectivity term (β={BETA}): {connectivity_term:.2f}")
        print(f"  Disconnection penalty (γ={GAMMA}): {disconnection_penalty:.2f}")
        print(f"  Obstacle penalty (ζ={ZETA}): {obstacle_penalty:.2f}")
        print(f"  Total objective: {objective:.2f}")
    
    return objective


def visualize_coverage(visited_unexplored, path_array):
    """Visualize the coverage map with visited cells and robot positions"""
    print("\nVisualizing Coverage Map...")
    
    # Get final positions
    final_pos = {}
    for r in range(R):
        final_position = tuple(path_array[r][-1])
        final_pos[final_position] = r + 1
    
    print("\n" + "="*50)
    print(f"Coverage Map ({N}x{M})")
    print("="*50)
    
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
            elif Map[i, j] == 2:
                print(" #", end=" ")
            else:
                print(" .", end=" ")
        print()
    
    # Print stats
    print("="*50)
    print(f"Cells explored: {len(visited_unexplored)}/{N*M} ({len(visited_unexplored)*100/(N*M):.1f}%)")
    print(f"R* = Final robot position, * = Explored, # = Obstacle, . = Unexplored")
    print("="*50 + "\n")
    

if __name__ == "__main__":
    # Create dummy solution
    path_array = create_dummy_solution()
    print("Solution:")
    print(path_array)
    
    # Check feasibility
    print(f"Is solution feasible? {is_feasible(path_array)}")
    
    # Calculate cost with visualization
    cost = cost_function(path_array, visualize=True)
    print(f"\nFinal Cost: {cost:.2f}")