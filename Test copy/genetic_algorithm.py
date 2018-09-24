#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Ben Wooldridge
Class: TE15
Date created: 2018-05-10
Comments: https://drive.google.com/file/d/108wd7DU3NtjuimirGHUgL_s3wb6FA7WA/view?usp=sharing
Requirements:
"""

from random import randint, choice
from math import ceil
from heapq import nlargest
import pickle
from neural_network_2 import SnakeBrain

# Global variables, placed them here instead of in the class just because these might be interesting to tinker with
population_size = 500
# How many individuals to select for breeding
# The amount of offspring from per parent is: (populations_size - keep_per_gen - freaks_per_gen) // parents_per_gen
parent_pairs_per_gen = 35
# How many of the best individuals to carry over to the next generation
keep_per_gen = 2
# How many new, completely random individuals to add each generation, seperate from the mutations
freaks_per_gen = 15

mutation_chance_percentage = 8

use_uniform_crossover = True
use_tournament_selection = False


# Game details
show_graphics = False
spawn_n_fruits = 5
delay = 0

# Used for saving a unique snake, not a good solution but works for the time-being
snakenum = randint(0, 999)

print("Population size: {}\n"
      "Parent pairs per generation: {}\n"
      "Keep per generation: {}\n"
      "Freaks per generation: {}\n"
      "Crossover method: {}\n"
      "Mutation chance: {}%\n======================================"
      .format(population_size,
              parent_pairs_per_gen,
              keep_per_gen,
              freaks_per_gen,
              "Uniform" if use_uniform_crossover else "Single-Point",
              mutation_chance_percentage))


class GeneticAlgorithm:
    def __init__(self):
        self.pop = Population()

        self.converged = False

        self.generation = 1

        self.top_score = 0

        latest_highest_fitnesses = []

        while not self.converged:

            # Play the game with every brain in the population, and save their fitness
            self.pop.calc_fitness()

            # Save the fittest score of the current generation, only used for logging
            self.pop.calc_fittest_score()

            latest_highest_fitnesses.append(self.pop.highest_fitness)
            if len(latest_highest_fitnesses) > 10:
                latest_highest_fitnesses.pop(0)

            # This is the new population we will be replacing the old one with
            new_pop = []

            mutated = 0

            for _ in range(parent_pairs_per_gen):

                # Select two parents, where higher fitnesses equate to a higher chance to be selected
                parent1 = self.pop.fitness_based_selection() if not use_tournament_selection else \
                          self.pop.tournament_selection()

                parent2 = self.pop.fitness_based_selection() if not use_tournament_selection else \
                    self.pop.tournament_selection()

                # Create n children per parent pair, to keep a stable population size. It won't always be the start size
                for n in range(int(ceil((len(self.pop.population) - keep_per_gen - freaks_per_gen) /
                                        (parent_pairs_per_gen * 2)))):

                    # Create child from two chosen snakes, using the chosen crossover method
                    child_snake1, child_snake2 = \
                        parent1.single_point_crossover(parent2) \
                        if not use_uniform_crossover \
                        else parent1.uniform_crossover(parent2)

                    # % Chance of mutation
                    if randint(1, 100) <= mutation_chance_percentage:
                        # Select a child at random to mutate
                        choice([child_snake1, child_snake2]).mutate()
                        mutated += 1

                    new_pop.append(child_snake1)
                    new_pop.append(child_snake2)

            # Select top individuals to be directly carried over to next generation, from a sorted population
            for snake in sorted(self.pop.population, key=lambda e: e[1], reverse=True)[:keep_per_gen]:
                new_pop.append(snake[0])

            # Generate a few completely random new snakes
            for i in range(freaks_per_gen):
                new_pop.append(SnakeBrain())

            print("Generation: {}  ||  Highest fitness: {}  || Average highest last 10:  {}  ||  Mutated children: {}".
                  format(str(self.generation).rjust(5),
                         str(self.pop.highest_fitness).rjust(5),
                         str(sum(latest_highest_fitnesses) / 10)[:5].rjust(5),
                         str(mutated).rjust(5)))

            # Save fit snake for testing
            if self.pop.highest_fitness > self.top_score:
                with open("fittest_snake_" + str(snakenum) + ".pickle", "wb") as snake:

                    meta_info = {
                        "snake_settings":
                            "Population size: {}\n"
                            "Parent pairs per generation: {}\n"
                            "Keep per generation: {}\n"
                            "Freaks per generation: {}\n"
                            "Crossover method: {}\n"
                            "Mutation chance: {}%\n"
                                .format(population_size,
                                        parent_pairs_per_gen,
                                        keep_per_gen,
                                        freaks_per_gen,
                                        "Uniform" if use_uniform_crossover else "Single-Point",
                                        mutation_chance_percentage),
                        "n_fruits": spawn_n_fruits,
                        "snake": self.pop.population[self.pop.fittest_index][0],
                        "score": self.pop.highest_fitness
                    }

                    pickle.dump(meta_info, snake)
                    self.top_score = self.pop.highest_fitness
                    print("Saved snake", str(snakenum) + "!")

            # Create a new population with the new and improved children of the old parents
            self.pop = Population(pop_size=0, existing_brains=new_pop)

            self.generation += 1


class Population:
    def __init__(self, pop_size=population_size, existing_brains=list()):

        self.population = []

        # Populate population with new snek brains
        if not existing_brains:
            for _ in range(pop_size):
                # Append a brain, and its fitness. Fitnesses start at zero.
                # A pair of [SnakeBrain, Fitness] will be reffered to as 'snake'
                self.population.append([SnakeBrain(), 0])
        # Reset the fitness of all the old snek brains, and add them to the population
        else:
            for brain in existing_brains:
                self.population.append([brain, 0])

        # Just for logging
        self.highest_fitness = 0
        self.fittest_index = 0

    def calc_fitness(self):
        """
        This is where we tell the brains to play the game, and save their score as their fitness
        :return:
        """

        for i, snake in enumerate(self.population):
            # Returns the score of the brains game
            score, age = snake[0].play(graphical=show_graphics, delay=delay, fruits=spawn_n_fruits)
            # The main fitness function
            fitness = age * (2**score)

            self.population[i][1] = fitness  # Update the brains fitness in the population list

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

        self.highest_fitness = max_fit

        # print("Fittest:", sorted(self.population, key=lambda e: e[1], reverse=True)[0][0].id)
        # print("HF:", self.highest_fitness)
        # print("HI:", self.fittest_index)
        # print(self.population[self.fittest_index][0].id)

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
            if current_sum >= r:
                # Return brain of chosen snake
                return snake[0]
        print("==== FAILED ====")
        print("Sum:", sum_fitnesses)
        print("Point", r)
        print("Current:", current_sum)
        return self.population[0][0]

    def tournament_selection(self, size=2):

        candidates = [choice(self.population) for _ in range(size)]

        winners = nlargest(2, candidates)  # todo

        return winners



GeneticAlgorithm()
