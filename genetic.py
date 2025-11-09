from optimization.base_optimizer import BaseOptimizer
from main import (
    movements_to_positions,
    cost_function,
    ROBOTS_POSITIONS,
)

class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for multi-robot path planning.
    
    Inherits from BaseOptimizer and implements the genetic algorithm specific methods.
    """
    # NOTE: We deduce the crossover size from the remainder of population size after elite and mutation
    def __init__(self, population_size=100, generation_size=100, mutation_rate=0.1, elite_rate=0.1, visualizer=None):
        """
        Initialize the Genetic Algorithm optimizer.
        
        Args:
            population_size (int): Number of individuals in the population
            mutation_rate (float): Probability of mutation
            crossover_rate (float): Probability of crossover
            **kwargs: Additional arguments for BaseOptimizer
        """
        super().__init__(max_iterations=generation_size, visualizer=visualizer)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
    
    # TODO: Still needs to handle logic where if an obstacle is encountered, the path is regenerated for rest of K for all robots.
    # TODO: Make sure to generate initial population that is feasible.
    def run(self, initial_population):
        """
        Perform genetic algorithm optimization.
        
        Implementation of abstract method from BaseOptimizer.
        
        Args:
            initial_population: Initial population of solutions
            
        Returns:
            tuple: (best_solution, best_cost)
        """

        # Track best solution across all generations
        best_solution = None
        best_cost = float('inf')
        
        # Main GA loop for generation_size iterations
        population = initial_population
        for generation_number in range(self.max_iterations):

            # Evaluate fitness for all individuals in the population
            fitness_list = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                # Store fitness values
                fitness_list.append(fitness)

            # Select current best solution
            current_cost = min(fitness_list)
            best_index = fitness_list.index(current_cost)
            current_solution = population[best_index]
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution

            
            new_population = self.generate_neighbor(population, fitness_list)
        

            population = new_population
        
            self.update_visualization(
                iteration=generation_number,
                current_cost=current_cost,
                best_cost=best_cost,
                best_path=movements_to_positions(best_solution, ROBOTS_POSITIONS),
                current_path=movements_to_positions(current_solution, ROBOTS_POSITIONS)
            )
            
            # Wait for visualization using base class method
            self.wait_for_visualization()
            
            if generation_number % 10 == 0:
                print(f"Generation {generation_number:4d} | Current: {current_cost:7.6f} | Best: {best_cost:7.6f}")

        # Notify visualization that optimization is complete using base class method
        self.finish_optimization()

        print("\n--- Optimization Complete ---")
        print(f"Best cost found: {best_cost:.6f}")
        return best_solution, best_cost

    def get_hyperparameters(self):
        """
        Get SA-specific hyperparameters.
        
        Overrides base class method to include SA-specific parameters.
        
        Returns:
            dict: Dictionary of hyperparameter names and values
        """
        params = super().get_hyperparameters()
        params.update({
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'elite_rate': self.elite_rate
        })
        return params

    def generate_neighbor(self, solution, fitness_list):
        """
        Generate a neighboring population of solutions using genetic operations.
        
        This method should implement crossover and mutation specific to
        the genetic algorithm.
        
        Args:
            solution: Current population
            
        Returns:
            A neighboring solution
        """

        # Create new population
        new_population = []

        # Choose survivors (elites) to carry over to next generation
        elites = self.select_survivors(self, fitness_list, solution)
        
        new_population.extend(elites)


        # Apply mutation to worst individuals
        worst_individuals = self.select_individuals_for_mutation(self, fitness_list, solution)

        for individual in worst_individuals:
            mutated_individual = self.mutate(individual)
            new_population.append(mutated_individual)
        
    
        # Create offspring through crossover
        num_crossover = self.population_size - len(new_population)
        offspring_count = 0
        # TODO: choose the best offspring
        while offspring_count < num_crossover:
            # Select parents
            parent1, parent2 = self.select_parents(solution, fitness_list, 2)
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.append(offspring1)
            offspring_count += 1
            if offspring_count < num_crossover:
                new_population.append(offspring2)
                offspring_count += 1

        return new_population

    def evaluate_fitness(self, solution, robots_positions=ROBOTS_POSITIONS):
        """
        Evaluate the fitness of a solution.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            float: Fitness value (lower is better for minimization)
        """
        # Calculate objective function value
        # TODO: The ROBOT Positions would need to be passed here for implementation of logic related to regeneration of path when an obstacle is encountered.
        path = movements_to_positions(solution, robots_positions)
        fitness = cost_function(path)
        
        # Return fitness value
        return fitness

    def sort_population_by_fitness(self, fitnesses, population):
        """
        Sort population based on fitness values.
        
        Args:
            fitnesses: Fitness values of current population
            population: Current population
            
        Returns:
            sorted_population: Population sorted by fitness (best first)
        """
        # Sort population by fitness values (lower is better)
        sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
        return sorted_population

    def select_survivors(self, fitnesses, population):
        """
        Select survivors for the next generation.
        
        Args:
            fitnesses: Fitness values of current population
            population: Current population
            
        Returns:
            survivors: Selected individuals for next generation
        """
        # Uses Survival of the Fittest strategy

        # Elitism: carry over top elite_rate individuals to new population
        num_elites = int(self.elite_rate * self.population_size)
        # Sort population by fitness and select elites
        sorted_population = self.sort_population_by_fitness(fitnesses, population)
        return sorted_population[:num_elites]
    
    def select_individuals_for_mutation(self, fitnesses, population):
        """
        Select individuals for mutation based on fitness.
        
        Args:
            fitnesses: Fitness values of current population
            population: Current population
        Returns:
            list: Selected individuals for mutation
        """        
        # Select worst individuals based on fitness for mutation
        num_to_mutate = int(self.mutation_rate * self.population_size)
        sorted_population = self.sort_population_by_fitness(fitnesses, population)
        return sorted_population[-num_to_mutate:]



    # TODO: Acceptance Criterion Method (Already used in run() -> may be redundant in GA)
    def acceptance_criterion(self, current_cost, new_cost):
        """
        Determine if new solution should be accepted.
        
        For GA, this is typically based on fitness comparison during survivor selection.
        
        Args:
            current_cost: Cost of current solution
            new_cost: Cost of new solution
            
        Returns:
            bool: True if new solution should be accepted
        """
        # In GA, typically accept if new_cost is better (lower for minimization)
        # This method might not be heavily used in standard GA as selection is done via survivor selection
        pass
    


    # TODO: Crossover Methods --> Should check for `is_feasible(path/offspring)` after generating offspring, if not feasible, regenerate or repair.
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parent solutions.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            tuple: (offspring1, offspring2) or single offspring depending on implementation
        """
        # Apply crossover with probability crossover_rate
        
        # Implement crossover strategy (single-point, two-point, uniform, etc.)
        
        # Ensure offspring are valid solutions
        
        # Return offspring
        pass

        # TODO
    
    # Select parents using selection method (tournament, roulette, etc.)
    def select_parents(self, population, fitnesses, num_parents):
        """
        Select parent solutions for breeding.
        
        Args:
            population: Current population of solutions
            fitnesses: Fitness values for each solution
            num_parents: Number of parents to select
            
        Returns:
            list: Selected parent solutions
        """
        # Implement selection strategy (tournament, roulette wheel, rank-based, etc.)
        
        # Return selected parents
        pass

    def order_crossover(self, parent1, parent2):
        """
        Order Crossover (OX) for permutation problems.
        
        Preserves the relative order of elements from parents.
        
        Args:
            parent1: First parent permutation
            parent2: Second parent permutation
            
        Returns:
            tuple: (offspring1, offspring2)
        """
        # Select two random crossover points
        
        # Copy segment between crossover points from parent1 to offspring1
        
        # Fill remaining positions with elements from parent2 in order, skipping duplicates
        
        # Repeat process with roles reversed for offspring2
        
        # Return both offspring
        pass

    def partially_mapped_crossover(self, parent1, parent2):
        """
        Partially Mapped Crossover (PMX) for permutation problems.
        
        Creates a mapping between two segments and applies it to fill remaining positions.
        
        Args:
            parent1: First parent permutation
            parent2: Second parent permutation
            
        Returns:
            tuple: (offspring1, offspring2)
        """
        # Select two random crossover points
        
        # Copy segment between crossover points from parent1 to offspring1
        
        # Create mapping between corresponding elements in the crossover segments
        
        # For positions outside segment, copy from parent2 and resolve conflicts using mapping
        
        # Repeat process with roles reversed for offspring2
        
        # Return both offspring
        pass

    def cycle_crossover(self, parent1, parent2):
        """
        Cycle Crossover (CX) for permutation problems.
        
        Identifies cycles in the parent permutations and alternates copying them.
        
        Args:
            parent1: First parent permutation
            parent2: Second parent permutation
            
        Returns:
            tuple: (offspring1, offspring2)
        """
        # Initialize offspring as copies of parents
        
        # Identify cycles by following position mappings between parents
        
        # Alternate copying cycles from parent1 and parent2 to offspring1
        
        # Swap parent roles for offspring2
        
        # Return both offspring
        pass

    def edge_recombination_crossover(self, parent1, parent2):
        """
        Edge Recombination Crossover for permutation problems.
        
        Preserves edges (adjacency relationships) from both parents.
        
        Args:
            parent1: First parent permutation
            parent2: Second parent permutation
            
        Returns:
            Solution with preserved edge information
        """
        # Build adjacency table storing neighbors for each element in both parents
        
        # Start with random element (or first element from a parent)
        
        # Repeatedly select next element with fewest unused neighbors
        
        # Remove selected element from all adjacency lists
        
        # Continue until all elements are included
        
        # Return offspring
        pass



    # TODO: Mutation Methods --> Should check for `is_feasible(path/offspring)` after generating offspring, if not feasible, regenerate or repair.

    def mutate(self, solution):
        """
        Apply mutation to a solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        # Apply mutation with probability mutation_rate
        
        # Implement mutation strategy (swap, insert, inversion, etc.)
        
        # Ensure mutated solution is valid
        
        # Return mutated solution
        pass

    def swap_mutation(self, solution):
        """
        Swap Mutation: randomly swap two positions in the permutation.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select two random distinct positions
        
        # Swap elements at those positions
        
        # Return mutated solution
        pass

    def insert_mutation(self, solution):
        """
        Insert Mutation: remove an element and insert it at a different position.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select random position to remove element from
        
        # Select random position to insert element at
        
        # Remove element and insert at new position, shifting others
        
        # Return mutated solution
        pass

    def inversion_mutation(self, solution):
        """
        Inversion Mutation: reverse the order of a subsequence.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select two random positions defining a subsequence
        
        # Reverse the order of elements in that subsequence
        
        # Return mutated solution
        pass

    def scramble_mutation(self, solution):
        """
        Scramble Mutation: randomly shuffle elements in a subsequence.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select two random positions defining a subsequence
        
        # Randomly shuffle elements within that subsequence
        
        # Return mutated solution
        pass

    def displacement_mutation(self, solution):
        """
        Displacement Mutation: remove a subsequence and insert it at another position.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select two positions defining a subsequence to remove
        
        # Remove the subsequence
        
        # Select random position to insert the subsequence
        
        # Insert subsequence at new position
        
        # Return mutated solution
        pass

    def two_opt_mutation(self, solution):
        """
        2-opt Mutation: remove two edges and reconnect in the only other way.
        
        Common in TSP, reverses a segment to eliminate edge crossings.
        
        Args:
            solution: Solution permutation to mutate
            
        Returns:
            Mutated solution
        """
        # Select two random positions i and j where i < j
        
        # Reverse the subsequence between positions i and j
        # This is equivalent to removing edges (i-1,i) and (j,j+1)
        # and adding edges (i-1,j) and (i,j+1)
        
        # Return mutated solution
        pass

    def adaptive_mutation(self, solution, generation, max_generations):
        """
        Adaptive Mutation: adjust mutation intensity based on convergence.
        
        Args:
            solution: Solution permutation to mutate
            generation: Current generation number
            max_generations: Maximum number of generations
            
        Returns:
            Mutated solution
        """
        # Calculate adaptive mutation rate based on generation progress
        # Early generations: more aggressive mutations (larger subsequences)
        # Later generations: fine-tuning mutations (smaller changes)
        
        # Select mutation type based on adaptive rate
        
        # Apply selected mutation
        
        # Return mutated solution
        pass