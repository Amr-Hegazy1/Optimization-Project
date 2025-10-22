"""
Configuration file for Multi-Robot Path Planning Optimization.

Modify these parameters to customize the optimization behavior.
"""

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

# Map dimensions
MAP_WIDTH = 20
MAP_HEIGHT = 20

# Robot configuration
ROBOT_INITIAL_POSITIONS = [
    (5, 0),   # Robot 1
    (0, 5),   # Robot 2
    (0, 15),  # Robot 3
    (5, 19),  # Robot 4
]

# Path length (number of steps)
PATH_LENGTH = 130


# ============================================================================
# CONSTRAINT PARAMETERS
# ============================================================================

# Maximum cells each robot can traverse
ENERGY_BUDGET = 100

# Communication radius for robot-to-robot links
COMMUNICATION_RADIUS = 15.0

# Threshold distance for network connectivity
CONNECTIVITY_THRESHOLD = 15.0


# ============================================================================
# OBJECTIVE FUNCTION WEIGHTS
# ============================================================================

# Coverage weight (higher = prioritize exploration)
# Used as: α / Coverage
ALPHA = 300 

# Connectivity weight (higher = prioritize staying connected)
# Used as: β / Connectivity
BETA = 100 

# Disconnection penalty weight (higher = stronger penalty for network splits)
# Used as: γ * P_disconnect
GAMMA = 4.0 

# Note: Obstacle avoidance is handled by hard constraints in is_feasible()
# ZETA parameter has been removed as obstacles are now strictly forbidden


# ============================================================================
# SIMULATED ANNEALING PARAMETERS
# ============================================================================

# Initial temperature for SA
SA_INITIAL_TEMPERATURE = 10.0 # to explore more at the start

# Cooling rate (0 < rate < 1, closer to 1 = slower cooling)
SA_COOLING_RATE = 0.995 

# Minimum temperature threshold for stopping
SA_MIN_TEMPERATURE = 0.1 

# Maximum number of iterations
SA_MAX_ITERATIONS = 5000




# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Animation step size (1 = show every step, higher = skip frames for faster animation)
VISUALIZATION_STEP_SIZE = 1000

# Initial animation speed in milliseconds per frame
ANIMATION_SPEED_MS = 50


# ============================================================================
# MAP INITIALIZATION
# ============================================================================

# Map cell types
CELL_UNEXPLORED = 0
CELL_FREE = 1
CELL_OBSTACLE = 2
CELL_ROBOT = 3

# Movement deltas: Up, Down, Left, Right, Stay
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
