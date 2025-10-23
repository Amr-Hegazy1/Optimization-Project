"""
Base class for optimization algorithms.

All optimization techniques should inherit from this class and implement
the required abstract methods to ensure a consistent interface.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    This class defines the interface that all optimization algorithms must follow,
    ensuring consistency across different optimization techniques (SA, GA, PSO, etc.).
    
    Attributes:
        max_iterations (int): Maximum number of optimization iterations
        visualizer: Optional visualization object for real-time updates
    """
    
    def __init__(self, max_iterations=5000, visualizer=None):
        """
        Initialize the base optimizer.
        
        Args:
            max_iterations (int): Maximum number of iterations
            visualizer: Optional visualizer object that implements update methods
        """
        self.max_iterations = max_iterations
        self.visualizer = visualizer
        self.current_iteration = 0
        self.best_solution = None
        self.best_cost = np.inf  # Initialize to infinity for minimization
        
        # History tracking for fast mode replay
        self.optimization_history = []
        self.fast_mode = False
        
    @abstractmethod
    def run(self, initial_solution):
        """
        Run the optimization algorithm.
        
        Args:
            initial_solution: Initial solution to start optimization from
            
        Returns:
            tuple: (best_solution, best_cost)
        """
        pass
    
    @abstractmethod
    def generate_neighbor(self, solution):
        """
        Generate a neighboring solution.
        
        This method should implement the neighborhood structure specific to
        the optimization algorithm.
        
        Args:
            solution: Current solution
            
        Returns:
            A neighboring solution
        """
        pass
    
    @abstractmethod
    def acceptance_criterion(self, current_cost, new_cost):
        """
        Determine whether to accept a new solution.
        
        Different algorithms have different acceptance criteria:
        - SA: probabilistic based on temperature
        - Hill Climbing: only accept improvements
        - GA: selection based on fitness
        
        Args:
            current_cost (float): Cost of current solution
            new_cost (float): Cost of candidate solution
            
        Returns:
            bool: True if new solution should be accepted
        """
        pass
    
    def update_visualization(self, **kwargs):
        """
        Update the visualization with current optimization state.
        
        This method should be called during optimization to provide real-time
        feedback. Subclasses can override to provide additional parameters.
        
        Args:
            **kwargs: Algorithm-specific visualization parameters
        """
        if self.fast_mode:
            # In fast mode, store history instead of updating visualization
            self.optimization_history.append(kwargs.copy())
        elif self.visualizer is not None:
            self.visualizer.update_optimization(**kwargs)
    
    def wait_for_visualization(self):
        """
        Wait for visualization to complete current animation.
        
        This ensures synchronization between optimization and visualization.
        """
        if self.fast_mode:
            # In fast mode, don't wait for visualization
            return
        if self.visualizer is not None and hasattr(self.visualizer, 'wait_for_animation_complete'):
            self.visualizer.wait_for_animation_complete()
    
    def finish_optimization(self):
        """
        Signal that optimization is complete.
        
        This allows the visualizer to update its state and display final results.
        """
        if self.fast_mode and self.visualizer is not None:
            # In fast mode, replay the optimization history
            if hasattr(self.visualizer, 'replay_optimization_history'):
                self.visualizer.replay_optimization_history(self.optimization_history)
        elif self.visualizer is not None and hasattr(self.visualizer, 'finish_optimization'):
            self.visualizer.finish_optimization()
    
    def get_algorithm_name(self):
        """
        Get the name of the optimization algorithm.
        
        Returns:
            str: Name of the algorithm (e.g., "Simulated Annealing", "Genetic Algorithm")
        """
        return self.__class__.__name__
    
    def get_hyperparameters(self):
        """
        Get algorithm-specific hyperparameters as a dictionary.
        
        Subclasses should override this to return their specific parameters.
        
        Returns:
            dict: Dictionary of hyperparameter names and values
        """
        return {
            'max_iterations': self.max_iterations
        }
