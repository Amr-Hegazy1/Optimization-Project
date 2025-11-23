"""
Configuration file for case studies.
Define different case study scenarios here.
"""

# Map configuration (copied from main config)
MAP_WIDTH = 20
MAP_HEIGHT = 20

# Robot initial positions (copied from main config)
ROBOT_INITIAL_POSITIONS = [
    (5, 0),   # Robot 1
    (0, 5),   # Robot 2
    (0, 15),  # Robot 3
    (5, 19),  # Robot 4
    (19, 5),  # Robot 5
    (15, 19)  # Robot 6
]

# Movement deltas
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

# ============================================================================
# CASE STUDY DEFINITIONS
# ============================================================================

# Case Study 1: (6 robots, 125 steps)
CASE_STUDY_1 = {
    'name': 'case_study_1',
    'num_robots': 6,
    'path_length': 125,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 125,
    'communication_radius': 15.0,
    'connectivity_threshold': 15.0,
    # Common iteration parameter for all algorithms
    'max_iterations': 200,
    # SA parameters
    'sa_initial_temp': 10.0,
    'sa_cooling_rate': 0.995,
    'sa_min_temp': 0.1,
    # GA parameters
    'ga_population': 20,
    'ga_mutation_rate': 0.3,
    'ga_elite_rate': 0.1,
    # ACO parameters
    'aco_num_ants': 30,
    'aco_alpha': 0.5,
    'aco_beta': 0.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}

# Case Study 2:  (4 robots, 100 steps)
CASE_STUDY_2 = {
    'name': 'case_study_2',
    'num_robots': 4,
    'path_length': 100,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 100,
    'communication_radius': 15.0,
    'connectivity_threshold': 15.0,
    # Common iteration parameter for all algorithms
    'max_iterations': 150,
    # SA parameters
    'sa_initial_temp': 10.0,
    'sa_cooling_rate': 0.995,
    'sa_min_temp': 0.1,
    # GA parameters
    'ga_population': 20,
    'ga_mutation_rate': 0.3,
    'ga_elite_rate': 0.1,
    # ACO parameters
    'aco_num_ants': 25,
    'aco_alpha': 0.5,
    'aco_beta': 0.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}

# Case Study 3: (3 robots, 50 steps)
CASE_STUDY_3 = {
    'name': 'case_study_3',
    'num_robots': 3,
    'path_length': 50,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 50,
    'communication_radius': 15.0,
    'connectivity_threshold': 15.0,
    # Common iteration parameter for all algorithms
    'max_iterations': 100,
    # SA parameters
    'sa_initial_temp': 10.0,
    'sa_cooling_rate': 0.995,
    'sa_min_temp': 0.1,
    # GA parameters
    'ga_population': 20,
    'ga_mutation_rate': 0.3,
    'ga_elite_rate': 0.1,
    # ACO parameters
    'aco_num_ants': 20,
    'aco_alpha': 0.5,
    'aco_beta': 0.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}

# Default case study to run
CHOSEN_CASE_STUDY = CASE_STUDY_2
