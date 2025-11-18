from optimization.base_optimizer import BaseOptimizer
from optimization.base_visualizer import BaseVisualizer
import numpy as np
from config import *

class AntColonyOptimizer(BaseOptimizer):
    
    def __init__(self, max_iterations: int = 5000, visualizer: BaseVisualizer = None, alpha: float = 1.0, beta: float = 0.0, evaporation_rate: float = 0.5, num_ants: int = 10) -> None:
        super().__init__(max_iterations, visualizer)
        self.alpha = alpha  # Importance of pheromone
        self.beta = beta    # Importance of heuristic information
        self.evaporation_rate = evaporation_rate  # Pheromone evaporation rate
        self.num_ants = num_ants  # Number of ants in the colony
        
        self.ants = np.zeros((self.num_ants, len(ROBOT_INITIAL_POSITIONS), PATH_LENGTH), dtype=int)  
        
        self.pheromone_map = np.ones((MAP_WIDTH, MAP_HEIGHT))  # Initial pheromone levels
        
        
    def _calculate_transition_probabilities(self, x, y) -> np.ndarray:
        
        probabilities = np.array([])
        sum_pheromone = 0
        for move in MOVES:
            new_x = x + move[0]
            new_y = y + move[1]
            
            if not (0 <= new_x < MAP_HEIGHT and 0 <= new_y < MAP_WIDTH):
                continue
            
            
            
            sum_pheromone += (self.pheromone_map[new_x, new_y] ** self.alpha)
        
        for move in MOVES:
            new_x = x + move[0]
            new_y = y + move[1]
            
            if not (0 <= new_x < MAP_HEIGHT and 0 <= new_y < MAP_WIDTH):
                probabilities = np.append(probabilities, 0)
                continue
            
            probability = (self.pheromone_map[new_x, new_y] ** self.alpha) / sum_pheromone
            probabilities = np.append(probabilities, probability)
        
        return np.cumsum(probabilities)
            
        
        
            
        
        
        
        
        
    def run(self, initial_solution=None):
        for iteration in range(self.max_iterations):
            
            
            
            
            for ant in range(self.num_ants):
                for r in range(R):
                    current_pos = ROBOT_INITIAL_POSITIONS[r]
                    
                    for t in range(PATH_LENGTH):
                        cum_probabilities = self._calculate_transition_probabilities(int(current_pos[0]), int(current_pos[1]))
                        move_probs = cum_probabilities
                        random_prob = np.random.rand() + 1e-10
                        
                        # clamp to the nearest valid move
                        next_move_index = np.searchsorted(move_probs, random_prob)
                        
                        next_move = MOVES[next_move_index]
                        new_x = current_pos[0] + next_move[0]
                        new_y = current_pos[1] + next_move[1]
                        
                        self.ants[ant, r, t] = next_move_index
                        current_pos = (new_x, new_y)
        
            # Evaporate pheromones
            self.pheromone_map *= (1 - self.evaporation_rate)
            
            # deposit new pheromones based on ants' paths
            for ant in range(self.num_ants):
                seen_positions = set()
                for r in range(R):
                    for t in range(PATH_LENGTH):
                        current_pos = self.ants[ant, r, t]
                        x, y = current_pos
                        x, y = int(x), int(y)
                        if (x, y) not in seen_positions:
                            seen_positions.add((x, y))
                            
                            self.pheromone_map[x, y] += self.cost_function(self.ants[ant])
                            
        costs = [self.cost_function(ant) for ant in self.ants]
        
        best_ant_index = np.argmin(costs)
        best_path = self.ants[best_ant_index]
        best_path = np.array(best_path, dtype=int)
        
        return best_path
        
    def generate_neighbor(self, solution):
        return super().generate_neighbor(solution)
            
            
            
    def acceptance_criterion(self, current_cost, new_cost):
        return super().acceptance_criterion(current_cost, new_cost)
            