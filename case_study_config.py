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

# Case Study 4: Higher max_iterations (300)
CASE_STUDY_4 = {
    'name': 'case_study_4',
    'num_robots': 4,
    'path_length': 100,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 100,
    'communication_radius': 15.0,
    'connectivity_threshold': 15.0,
    # Common iteration parameter for all algorithms
    'max_iterations': 300,
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

# Case Study 5: Higher SA initial temperature
CASE_STUDY_5 = {
    'name': 'case_study_5',
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
    'sa_initial_temp': 20.0,
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

# Case Study 6: Larger GA population
CASE_STUDY_6 = {
    'name': 'case_study_6',
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
    'ga_population': 50,
    'ga_mutation_rate': 0.3,
    'ga_elite_rate': 0.1,
    # ACO parameters
    'aco_num_ants': 25,
    'aco_alpha': 0.5,
    'aco_beta': 0.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}

# Case Study 7: ACO with non-zero beta and AS strategy
CASE_STUDY_7 = {
    'name': 'case_study_7',
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
    'aco_beta': 1.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}

# Case Study 8: Different global alpha and beta
CASE_STUDY_8 = {
    'name': 'case_study_8',
    'num_robots': 4,
    'path_length': 100,
    'alpha': 2000,
    'beta': 300,
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

# Case Study 9: Smaller communication radius
CASE_STUDY_9 = {
    'name': 'case_study_9',
    'num_robots': 4,
    'path_length': 100,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 100,
    'communication_radius': 10.0,
    'connectivity_threshold': 10.0,
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

# Case Study 10: Fewer robots (2)
CASE_STUDY_10 = {
    'name': 'case_study_10',
    'num_robots': 2,
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

# Case Study 11: Longer path length (150)
CASE_STUDY_11 = {
    'name': 'case_study_11',
    'num_robots': 4,
    'path_length': 150,
    'alpha': 3000,
    'beta': 500,
    'gamma': 4.0,
    'energy_budget': 150,
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
    'aco_num_ants': 25,
    'aco_alpha': 0.5,
    'aco_beta': 0.0,
    'aco_evaporation': 0.4,
    'aco_strategy': 'as'
}
