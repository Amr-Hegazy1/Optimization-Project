from optimization.base_optimizer import BaseOptimizer
from optimization.base_visualizer import BaseVisualizer
import numpy as np
from config import *

class AntColonyOptimizer(BaseOptimizer):
    
    def __init__(self, max_iterations: int = 5000, visualizer: BaseVisualizer = None, alpha: float = 1.0, beta: float = 0.0, evaporation_rate: float = 0.4, num_ants: int = 10) -> None:
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
        # Initialize tracking variables
        best_cost = float('inf')
        best_solution = None
        
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
                ant_path = self.movements_to_positions(self.ants[ant])
                ant_cost = self.cost_function(ant_path)
                seen_positions = set()
                for r in range(R):
                    current_pos = ROBOT_INITIAL_POSITIONS[r]
                    for t in range(PATH_LENGTH):
                        move_index = self.ants[ant, r, t]
                        move = MOVES[move_index]
                        x = int(current_pos[0] + move[0])
                        y = int(current_pos[1] + move[1])
                        if (x, y) not in seen_positions:
                            seen_positions.add((x, y))
                            self.pheromone_map[x, y] += 1.0 / (1.0 + ant_cost)
                        current_pos = (x, y)
                        
            # Find best ant
            costs = []
            for ant in range(self.num_ants):
                ant_path = self.movements_to_positions(self.ants[ant])
                costs.append(self.cost_function(ant_path))
            
            current_best_ant_index = np.argmin(costs)
            current_cost = costs[current_best_ant_index]
            current_solution = self.ants[current_best_ant_index]
            
            # Update global best
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()
            
            # Update visualization if available
            if self.visualizer is not None:
                # Calculate pheromone statistics
                avg_pheromone = np.mean(self.pheromone_map)
                max_pheromone = np.max(self.pheromone_map)
                
                # Convert solutions to position paths for visualization
                current_path = self.movements_to_positions(current_solution)
                best_path = self.movements_to_positions(best_solution)
                
                # Store history for fast mode
                update_kwargs = {
                    'iteration': iteration,
                    'current_cost': current_cost,
                    'best_cost': best_cost,
                    'best_path': best_path,
                    'avg_pheromone': avg_pheromone,
                    'max_pheromone': max_pheromone,
                    'current_path': current_path
                }
                
                if self.fast_mode:
                    self.optimization_history.append(update_kwargs)
                else:
                    self.visualizer.update_optimization(**update_kwargs)
                    if hasattr(self.visualizer, 'wait_for_animation_complete'):
                        self.visualizer.wait_for_animation_complete()
        
        # Replay history in fast mode
        if self.fast_mode and self.visualizer is not None:
            self.visualizer.replay_optimization_history(self.optimization_history)
        
        # Finish visualization
        if self.visualizer is not None:
            if not self.fast_mode:
                self.visualizer.finish_optimization()
        
        best_path = self.movements_to_positions(best_solution) if best_solution is not None else None
        
        return best_path
        
    def generate_neighbor(self, solution):
        return super().generate_neighbor(solution)
            
            
            
    def acceptance_criterion(self, current_cost, new_cost):
        return super().acceptance_criterion(current_cost, new_cost)
            