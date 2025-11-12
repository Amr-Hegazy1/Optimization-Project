import random
from optimization.base_optimizer import BaseOptimizer
from main import (
    movements_to_positions,
    cost_function,
    is_feasible,
    ROBOTS_POSITIONS,
)


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for multi-robot path planning.

    Inherits from BaseOptimizer and implements the genetic algorithm specific methods.
    """

    # NOTE: We deduce the crossover size from the remainder of population size after elite and mutation
    def __init__(
        self,
        population_size=100,
        generation_size=100,
        mutation_rate=0.1,
        elite_rate=0.1,
        visualizer=None,
    ):
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
    def run(
        self,
        initial_population,
        mutation_method="swap",
        parent_selection_method="sus",
        crossover_method="one_point",
        robot_positions=ROBOTS_POSITIONS,
    ):
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
        best_cost = float("inf")

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

            new_population = self.generate_neighbor(
                population,
                fitness_list,
                mutation_method=mutation_method,
                parent_selection_method=parent_selection_method,
                crossover_method=crossover_method,
                robot_positions=robot_positions,
            )

            population = new_population

            self.update_visualization(
                iteration=generation_number,
                current_cost=current_cost,
                best_cost=best_cost,
                best_path=movements_to_positions(best_solution, ROBOTS_POSITIONS),
                current_path=movements_to_positions(current_solution, ROBOTS_POSITIONS),
            )

            # Wait for visualization using base class method
            self.wait_for_visualization()

            if generation_number % 10 == 0:
                print(
                    f"Generation {generation_number:4d} | Current: {current_cost:7.6f} | Best: {best_cost:7.6f}"
                )

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
        params.update(
            {
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "elite_rate": self.elite_rate,
            }
        )
        return params

    def acceptance_criterion(self, current_cost, new_cost):
        return super().acceptance_criterion(current_cost, new_cost)

    def generate_neighbor(
        self,
        solution,
        fitness_list,
        mutation_method="swap",
        parent_selection_method="SUS",
        crossover_method="one_point",
        robot_positions=ROBOTS_POSITIONS,
    ):
        """
        Generate a neighboring population of solutions using genetic operations.

        This method should implement crossover and mutation specific to
        the genetic algorithm.

        Args:
            solution: Current population

        Returns:
            A neighboring solution
        """

        # ! Take Care: if any non-random way for parent selection and crossover is used, this could lead to infinite loops if no feasible offspring is generated, when adding new crossover or parent selection methods please take care of that. Like for example if always selecting the top 2 fittest individuals as parents and a crossover method that does not produce feasible offsprings from those parents, then the while loop below may never end.

        # Create new population
        new_population = []

        # Choose survivors (elites) to carry over to next generation
        elites = self.select_survivors(self, fitness_list, solution)

        new_population.extend(elites)

        # Apply mutation to worst individuals
        worst_individuals = self.select_individuals_for_mutation(
            self, fitness_list, solution
        )

        for individual in worst_individuals:
            mutated_individual = self.mutate(individual, mutation_method=mutation_method)
            while not is_feasible(movements_to_positions(mutated_individual, robot_positions)):
                mutated_individual = self.mutate(individual)
            new_population.append(mutated_individual)

        # Create offspring through crossover
        num_crossover = self.population_size - len(new_population)
        offspring_count = 0
        while offspring_count < num_crossover:
            # Select parents
            # here we use random methods to ensure that if offspring are not feasible, we can try again with different parents
            # however if a parent selection method is used with no randomness, such as selecting the top 2 fittest individuals, this could lead to infinite loops
            # At least one of the parent_selection or crossover_methods should include some stochasticity to avoid any infinite loops
            parent1, parent2 = self.select_parents(
                solution,
                fitness_list,
                2,
                parent_selection_method=parent_selection_method,
            )
            offspring1, offspring2 = self.crossover(
                parent1, parent2, crossover_method=crossover_method
            )
            offsprings_sorted_by_fitness = sorted(
                [offspring1, offspring2], key=lambda x: self.evaluate_fitness(x)
            )

            # this ensures that we only add feasible offsprings to the new population
            for offspring in offsprings_sorted_by_fitness:
                if not is_feasible(movements_to_positions(offspring, robot_positions)):
                    offsprings_sorted_by_fitness.remove(offspring)

            if len(offsprings_sorted_by_fitness) != 0:
                # choose the fittest (lowest cost)
                new_population.append(offsprings_sorted_by_fitness[0])
                offsprings_sorted_by_fitness.remove(offsprings_sorted_by_fitness[0])
                offspring_count += 1
            else:
                # array is empty, then no offspring was feasible, so generate new parents and try another form of crossover, this depends that select_parents has some randomness to it
                continue
            if offspring_count < num_crossover:
                if len(offsprings_sorted_by_fitness) != 0:
                    new_population.append(offsprings_sorted_by_fitness[0])
                    offsprings_sorted_by_fitness.remove(offsprings_sorted_by_fitness[0])
                    offspring_count += 1
                else:
                    continue
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

    # Select parents using selection method (tournament, roulette, etc.)
    def select_parents(
        self, population, fitnesses, num_parents, parent_selection_method="SUS"
    ):
        """
        Select parent solutions for breeding.

        Args:
            population: Current population of solutions
            fitnesses: Fitness values for each solution
            num_parents: Number of parents to select

        Returns:
            list: Selected parent solutions
        """
        # Implementing Stochastic Universal Sampling (SUS) as an example
        if parent_selection_method == "sus":
            return self.select_parents_by_sus(population, fitnesses, num_parents)
        if parent_selection_method == "roulette":
            selected_parents = []
            for _ in range(num_parents):
                pointer = random.uniform(0, sum(fitnesses))
                parent = self.select_parent_by_roulette(population, fitnesses, pointer)
                selected_parents.append(parent)
            return selected_parents
        else:
            raise ValueError(
                f"Unknown parent selection method: {parent_selection_method}"
            )

    def select_parents_by_sus(self, population, fitnesses, num_parents):
        """
        Stochastic Universal Sampling (SUS) for parent selection.

        Args:
            population: Current population of solutions
            fitnesses: Fitness values for each solution
            num_parents: Number of parents to select
        Returns:
            list: Selected parent solutions
        """
        # Calculate total fitness
        total_fitness = sum(fitnesses)
        # Calculate distance between pointers
        pointer_distance = total_fitness / num_parents
        # Select a random start point
        start_point = random.uniform(0, pointer_distance)
        # Generate pointers
        pointers = [start_point + i * pointer_distance for i in range(num_parents)]

        selected_parents = []
        for pointer in pointers:
            parent = self.selected_parents_by_roulette(population, fitnesses, pointer)
            selected_parents.append(parent)
        return selected_parents

    # Selects a parent based on a single pointer using roulette wheel selection
    def select_parent_by_roulette(self, population, fitnesses, pointer):
        """
        Roulette Wheel Selection for a single parent.

        Args:
            population: Current population of solutions
            fitnesses: Fitness values for each solution
            pointer: Selection pointer
        Returns:
            Selected parent solution
        """
        cumulative_fitness = 0
        for individual, fitness in zip(population, fitnesses):
            cumulative_fitness += fitness
            if cumulative_fitness >= pointer:
                return individual
        raise ValueError("Pointer exceeds total fitness; check fitness values.")

    def crossover(self, parent1, parent2, crossover_method="one_point"):
        """
        Perform crossover between two parent solutions.

        Args:
            parent1: First parent solution
            parent2: Second parent solution

        Returns:
            tuple: (offspring1, offspring2) or single offspring depending on implementation
        """
        if crossover_method == "one_point":
            return self.one_point_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {crossover_method}")

    def one_point_crossover(self, parent1, parent2):
        """
        One-Point Crossover for permutation problems.

        Args:
            parent1: First parent permutation
            parent2: Second parent permutation
        Returns:
            tuple: (offspring1, offspring2)
        """
        length = len(parent1)
        crossover_point = random.randint(1, length - 1)

        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

        return offspring1, offspring2

    # TODO
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
    # TODO
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
    # TODO
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
    # TODO
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


    def mutate(self, solution, mutation_method='swap'):
        """
        Apply mutation to a solution.

        Args:
            solution: Solution to mutate

        Returns:
            Mutated solution
        """

        # Implement mutation strategy (swap, insert, inversion, etc.)
        if mutation_method == 'swap':
            return self.swap_mutation(solution)
        else:
            raise ValueError(
                f"Unknown mutation method: {mutation_method}"
            )

    def swap_mutation(self, solution):
        """
        Swap Mutation: randomly swap two positions in the permutation.

        Args:
            solution: Solution permutation to mutate

        Returns:
            Mutated solution
        """
        # Select two random distinct positions
        position_1 = random.randint(0, len(solution) - 1)
        position_2 = random.randint(0, len(solution) - 1)
        while position_1 == position_2:
            position_2 = random.randint(0, len(solution) - 1)
        # Swap elements at those positions
        solution[position_1], solution[position_2] = solution[position_2], solution[position_1]

        # Return mutated solution
        return solution

    # TODO
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
    # TODO
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
    # TODO
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
    # TODO
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
    # TODO
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
    # TODO
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
