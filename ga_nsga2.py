import numpy as np
import copy
import pandas as pd
import nsga2 as nsga

class Individual():

    def __init__(self, program, fitness):
        self.program = np.array(program)
        self.fitness = fitness


class GA():
    def __init__(self, ind_size, pop_size, max_gens, random_state, mut_rate, cross_rate, fitness_func):

        """
        Initializer for the GA class.
        We are assuming that the fitness function is a maximizing problem.

        Parameters
        ----------
        pop_size : int
            Number of individuals for use in the population.
        mx_gens : int
            Number of generations to evolve for. 
        """

        self.max_gens = max_gens
        self.pop_size = pop_size
        self.ind_size = ind_size
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.fitness_func = fitness_func
        self.rng = np.random.default_rng(random_state)
        self.population = []
        self.best_individual = None
        self.evaluated_individuals = pd.DataFrame(columns = ['individual','perf_fitness','fair_fitness'])

    def initialize_population(self):

        """Generates the population list of Individuals."""

        for _ in range(self.pop_size):
            self.population.append(Individual(2*self.rng.random(size=self.ind_size), None))

        self.population = self.evaluate_population(self.population)

    def mutation(self, ind):
        ''' With a probability of mut_rate, mutate the individual. '''
        _program = copy.deepcopy(ind.program)
        for i in range(len(_program)):
            if self.rng.random()<self.mut_rate:
                if self.rng.random()<0.5:
                    new_val = _program[i] + self.rng.random()
                else:
                    new_val = _program[i] - self.rng.random()
                _program[i] = min(max(new_val,0),2)
                    
        return Individual(_program, None)

    def crossover(self, ind1, ind2):
        _program1 = copy.deepcopy(ind1.program)
        _program2 = copy.deepcopy(ind2.program)
        
        if self.rng.random()<self.cross_rate:
            child = []
            for i in range(len(_program1)):
                if self.rng.random()<0.5:
                    child.append(_program1[i])
                else:
                    child.append(_program2[i])
        else:
            child = _program1
                    
        return Individual(child, None)
    
    def ep_lexicase_selection(self):  
        # Maximize all objectives
        # Adapted from https://deap.readthedocs.io/en/master/_modules/deap/tools/selection.html#selAutomaticEpsilonLexicase
        candidates = self.population
        cases = list(range(len(self.population[0].fitness)))
        self.rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = [x.fitness[cases[0]] for x in candidates]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            best_val_for_case = max(errors_for_this_case )
            min_val_to_survive = best_val_for_case - median_absolute_deviation
            candidates = [x for x in candidates if x.fitness[cases[0]] >= min_val_to_survive]
        return self.rng.choice(candidates)
    

    def evaluate_population(self, pop):

        """
        Sets the fitness of the individual passed in. Make sure higher fitness values are better.

        Parameters
        ----------
        pop : list[Individual] 
        """

        
        for individual in pop:
            individual.fitness = self.fitness_func(individual.program)
            # Update self.evaluated_individuals
            self.evaluated_individuals.loc[len(self.evaluated_individuals.index)] = {'individual':individual.program,'perf_fitness':individual.fitness[0],'fair_fitness':individual.fitness[1]}

        return pop
        # Updating the best individual
        #candidates = self.population
        #cases = list(range(len(self.population[0].fitness)))
    
        #while len(cases) > 0 and len(candidates) > 1:
            #Anil Made Changes: in the next line; changed from 'max' to 'min'
        #    best_val_for_case = min([x.fitness[cases[0]] for x in candidates])
        #    candidates = [x for x in candidates if x.fitness[cases[0]] == best_val_for_case]
        #    cases.pop(0)
        #self.best_individual = self.rng.choice(candidates)


    def step_optimize(self):

        """
        Progresses the optimization prcedure by a single iteration.
        """

        ### Parent Selection
        pop = self.population
        parent_ids = []
        scores = np.array([[ind.fitness[0], ind.fitness[1]] for ind in pop], dtype=np.float32)
        parent_cnt = 2*self.pop_size # Number of parents to select; 2*pop_size because of crossover

        # get the fronts and rank
        fronts, ranks = nsga.non_dominated_sorting(obj_scores=scores, weights=np.array([1, 1], dtype=np.float32))
        # make sure that the number of fronts is correct
        assert sum([len(f) for f in fronts]) == len(ranks)

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(scores, np.int32(2))

        # get parent_cnt number of parents
        for _ in range(parent_cnt):
            parent_ids.append(nsga.non_dominated_binary_tournament(rng=self.rng, ranks=ranks, distances=crowding_distance))

        # Offspring Generation
        offspring = []
        j = 0
        for i in range(self.pop_size):
            parent_a = pop[parent_ids[j]]
            parent_b = pop[parent_ids[j+1]]

            child = self.crossover(parent_a, parent_b)

            child = self.mutation(child)

            offspring.append(child)

            j += 2


        # Evaluate the offspring
        offspring = self.evaluate_population(offspring)

        # combine both the population and offspring scores
        offspring_scores = np.array([[ind.fitness[0], ind.fitness[1]] for ind in offspring], dtype=np.float32)
        all_scores = np.array(np.concatenate((scores, offspring_scores), axis=0), dtype=np.float32)

        # get the fronts and rank
        fronts, _ = nsga.non_dominated_sorting(obj_scores=all_scores, weights=np.array([1,1], dtype=np.float32))

        # get crowding distance for each solution
        crowding_distance = nsga.crowding_distance(all_scores, np.int32(2))

        # truncate the population to the population size with nsga ii
        survivor_ids = nsga.non_dominated_truncate(fronts, crowding_distance, self.pop_size)
        # make sure that the number of survivors is correct
        assert len(survivor_ids) == self.pop_size

        # combine the population and offspring
        candidates = pop + offspring

        # subset the candidates to only include the survivors
        new_pop = []

        for i in survivor_ids:
            # make sure we are within the bounds of the candidates
            assert 0 <= i < len(candidates)
            new_pop.append(candidates[i])

        self.population = new_pop
        

    
    def optimize(self):

        """
        Responsible for managing the optimization process.
        """

        # First generation
        print("Generation 1 started:")
        self.initialize_population()
        print("Generation 1 ended.")
        # Rest of the generations
        for gen in range(self.max_gens-1):
            print("Generation ", gen+2, " started:")
            self.step_optimize()
            print("Generation ", gen+2, " ended.")
