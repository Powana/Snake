#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Ben Wooldridge
Class: TE15
Date created: 2018-05-10
Comments: https://drive.google.com/file/d/108wd7DU3NtjuimirGHUgL_s3wb6FA7WA/view?usp=sharing
Requirements:
"""

from random import randint, random
from math import ceil
from neural_network import SnakeBrain

# Global variables, placed them here instead of in the class just because these might be interesting to tinker with
population_size = 10000
# How many individuals to select for breeding
# The amount of offspring from per parent is: (populations_size - keep_per_gen - freaks_per_gen) // parents_per_gen
parents_per_gen = 50
# How many of the best individuals to carry over to the next generation
keep_per_gen = 5
# How many new, completely random individuals to add each generation, seperate from the mutations
freaks_per_gen = 10

mutation_chance_percentage = 5
show_graphics = False


class GeneticAlgorithm:
    def __init__(self):
        self.pop = Population()

        self.converged = False

        self.generation = 1

        latest_highest_fitnesses = []

        while not self.converged:

            # Play the game with every brain in the population, and save their fitness
            self.pop.calc_fitness()

            # Save the fittest score of the current generation, only used for logging
            self.pop.calc_fittest_score()

            latest_highest_fitnesses.append(self.pop.fittest_score)
            if len(latest_highest_fitnesses) > 10:
                latest_highest_fitnesses.pop(0)

            # This is the new population we will be replacing the old one with
            new_pop = []

            mutated = 0
            # print(len(self.pop.population))
            # Select parents for next generation
            # parents = self.pop.roulette_select(num=parents_per_gen)

            # for i, _ in enumerate(parents[::2]):
            for _ in range(parents_per_gen // 2):

                # Select two parents, where higher fitnesses equate to a higher chance to be selected
                parent1 = self.pop.fitness_based_selection()
                parent2 = self.pop.fitness_based_selection()

                # Create n children per parent pair, to keep a stable population size. It won't always be the start size
                for n in range(int(ceil(
                        (len(self.pop.population) - keep_per_gen - freaks_per_gen) / parents_per_gen * 2))):

                    # Create child from two chosen snakes, the child gets a random amount of qualites from each parent
                    child_snake = parent1.crossover(parent2)

                    # % Chance of mutation
                    if randint(1, 100) <= mutation_chance_percentage:
                        child_snake.mutate()
                        mutated += 1

                    new_pop.append(child_snake)

            # Select top individuals to be directly carried over to next generation, this population has been sorted
            for snake in self.pop.population[:keep_per_gen]:
                new_pop.append(snake[0])

            # Generate a few completely random new snakes
            for snake in range(freaks_per_gen):
                new_pop.append(SnakeBrain())

            # Replace entire population with the new and improved children of the old parents, and give them fitness 0
            self.pop.population = list([[new_snake, 0] for new_snake in new_pop])

            print("Generation: {}  ||  Highest fitness: {}  || Average highest last 10:  {}  ||  Mutated children: {}".
                  format(str(self.generation).rjust(5),
                         str(self.pop.fittest_score).rjust(5),
                         str(sum(latest_highest_fitnesses) / 10)[:5].rjust(5),
                         str(mutated).rjust(5)))

            self.generation += 1


class Population:
    def __init__(self, pop_size=population_size):
        self.population = []

        # Populate population with snek brains
        for _ in range(pop_size):
            # Append a brain, and it's fitness. Fitnesses start at zero.
            # A pair of [SnakeBrain, Fitness] will be reffered to as 'snake'
            self.population.append([SnakeBrain(), 0])

        # just for logging
        self.fittest_score = 0
        self.fittest_index = 0

    def calc_fitness(self):
        """
        This is where we tell the brains to play the game, and save their score as their fitness
        :return:
        """

        for i, snake in enumerate(self.population):
            score, age = snake[0].play(graphical=show_graphics)  # Returns the score of the brains game
            # The main fitness function
            fitness = age * 2**score
            self.population[i][1] = fitness  # Update the brains fitness in the population list

        # Sort the population by fitness, todo: check if actually needed
        self.population = sorted(self.population, key=lambda e: e[1], reverse=True)

    def calc_fittest_score(self):
        """
        Set self.fittest to the score snake that has the highest fittness identical fitnesses
        the last snake will be selected
        :return:
        """
        max_fit = 0

        for i, snake in enumerate(self.population):
            if snake[1] >= max_fit:
                max_fit = snake[1]
                self.fittest_index = i

        self.fittest_score = max_fit

    def roulette_select(self, num):
        """ Roulette selection, implemented according to:
            <http://stackoverflow.com/questions/177271/roulette
            -selection-in-genetic-algorithms/177278#177278>
        """
        population = self.population
        fitnesses = list([snake[1] for snake in self.population])  # [1, 0, 0, 1 , 0, 3, 2, 0, 1]
        total_fitness = float(sum(fitnesses))  # 6
        rel_fitness = [f / total_fitness for f in fitnesses]  # [1/8, 0, 0, 1/8, 0, 3/8, 2/8, 0, 1/8]
        # Generate probability intervals for each individual
        probs = [sum(rel_fitness[:i + 1]) for i in range(len(rel_fitness))]  # [1/8, 1/8, 1/8, 2/8, 2/8, 5/8, 7/8, 7/8, 8/8]
        new_population = []
        m = []
        for _ in range(num):
            r = random()
            for (i, individual) in enumerate(population):
                if r <= probs[i]:
                    m.append(individual[1])
                    new_population.append(individual[0])
                    break
        from numpy import mean
        return new_population

    def fitness_based_selection(self):
        """
        A slection process that chooses a snake, where a snake with a higher fitness has a higher chance of being
        selected
        :return: The chosen snakes brain
        """
        sum_fitnesses = sum(list([snake[1] for snake in self.population]))

        # A random cutoff digit.
        r = randint(0, sum_fitnesses)

        current_sum = 0

        for snake in self.population:
            current_sum += snake[1]
            if current_sum > r:
                # Return brain of chosen snake
                return snake[0]


# todo: I think the problem is that the fittest snakes aren't being selected, or there's a problem with the game output,
# or that the initial values arent random enough, the snake basically never changes direction
GeneticAlgorithm()
