#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Ben Wooldridge
Class: TE15
Date created: 2018-05-10
Comments: <Lycka till babe>
Requirements:
"""

from time import sleep
import numpy as np
import random
import snake_custom


class SnakeBrain:
    """
    SnakeBrain is the neural network controlling a snake.
    Input nodes: 24. 3 for each direction from the snakes head, telling it what is that direction.
    Hidden layers: 2
    Nodes per layer: 8
    Output nodes: 4. 1 for each direction the snake can choose

    """
    def __init__(self, weights=None, biases=None, input_size=24, nodes_per_layer=8, output_size=4):

        # Create a brain with random weights and biases
        if not weights and not biases:
            """
            Initialize random (normalised) weights and biases, and add them to a 2D array.
            2D arrays are used so that we can change shape of our input to the shape of our output.
            I didn't figure this out myself, but I understand the logic behind it.
            Pseudocode for the shapes of the weights and layers would be:
                
                # Shapes of input, weights, and output
                i = 1x24
                wih = 24x8
                whh = 8x8
                who = 8x4
                o = 1x4
                
                # Feeding the input through the weights with multiplication, showing the shapes of the arrays
                inputs = 1x24
                hiddenLayer1 = 1x24 * 24x8 = 1x8
                hiddenLayer2 = 1x8 * 8x8 = 1x8
                outputs = 1x8 * 8x4 = 1x4
            
            It is worth noting that these 2D arrays could be matrices without any consequences, but I chose to use
            arrays instead
            """

            # The weights and biases between the inputs and hidden layer 1
            # Shape: input_size x nodes_per_layer == 24x8
            self.weights_input_hidden1 = np.array([np.random.
                                                  uniform(low=-1, size=nodes_per_layer) for _ in range(input_size)])
            self.biases_input_hidden1 = np.random.uniform(low=-1, size=input_size)

            # The weights and biases between hidden layer 1 and hidden layer 2
            # Shape: nodes_per_layer x nodes_per_layer == 8x8
            self.weights_hidden1_hidden2 = np.array([np.random.
                                                    uniform(low=-1, size=nodes_per_layer) for _ in range(nodes_per_layer)])
            self.biases_hidden1_hidden2 = np.random.uniform(low=-1, size=self.weights_input_hidden1.shape[1])

            # The weights and biases between hidden layer 2 and output layer
            # Shape: nodes_per_layer x output_size == 8x4
            self.weights_hidden2_output = np.array([np.random.
                                                   uniform(low=-1, size=output_size) for _ in range(nodes_per_layer)])
            self.biases_hidden2_output = np.random.uniform(low=-1, size=output_size)

        # Create a brain with existing weights and biases
        else:
            # Weights and biases are passed as lists of weights and biases for each layer
            self.weights_input_hidden1 = weights[0]
            self.biases_input_hidden1 = biases[0]

            self.weights_hidden1_hidden2 = weights[1]
            self.biases_hidden1_hidden2 = biases[1]

            self.weights_hidden2_output = weights[2]
            self.biases_hidden2_output = biases[2]

        # Placeholder for the game of Snake this brain will act upon
        self.game = None

    def get_output(self, input_array: np.ndarray):
        """
        Get output from input by feed forwarding it through the network

        :param input_array: The input to get an output from, should be an array of the inputs
        :return: an output array with 4 values of the shape 1x4
        """

        # Add biases then multiply by weights, input => h_layer_1, this is done opposite because the input can be zero
        h_layer_1_b = input_array + self.biases_input_hidden1
        h_layer_1_w = np.dot(h_layer_1_b, self.weights_input_hidden1)
        h_layer_1 = self.sigmoid(h_layer_1_w)  # Run the output through a sigmoid function

        # Multiply by weights then add biases, h_layer_1 => h_layer_2
        h_layer_2_w = np.dot(h_layer_1, self.weights_hidden1_hidden2)
        h_layer_2_b = h_layer_2_w + self.biases_hidden1_hidden2
        h_layer_2 = self.sigmoid(h_layer_2_b)

        # Multiply by weights then add biases, h_layer_2 => output
        output_w = np.dot(h_layer_2, self.weights_hidden2_output)
        output_b = output_w + self.biases_hidden2_output

        output = self.sigmoid(output_b)

        return output

    @staticmethod
    def sigmoid(x):
        # The sigmoid function basically just exaggerates the value
        return 1 / (1 + np.exp(x))

    def single_point_crossover(self, second_brain):
        """
        Crossover two brains using single point co, creating a child with a random amount of each it's parents qualities
        :param second_brain: the brain to crossover with
        :return: SnakeBrain child of this brain and the second_brain
        """

        # The weights and biases for the second snake
        s_w1 = second_brain.weights_input_hidden1
        s_b1 = second_brain.biases_input_hidden1

        s_w2 = second_brain.weights_hidden1_hidden2
        s_b2 = second_brain.biases_hidden1_hidden2

        s_w3 = second_brain.weights_hidden2_output
        s_b3 = second_brain.biases_hidden2_output

        # Childs weights and biases

        c1_w1, c2_w1 = self.__2d_array_sp_crossover(self.weights_input_hidden1, s_w1)
        c1_b1, c2_b1 = self.__1d_array_sp_crossover(self.biases_input_hidden1, s_b1)

        c1_w2, c2_w2 = self.__2d_array_sp_crossover(self.weights_hidden1_hidden2, s_w2)
        c1_b2, c2_b2 = self.__1d_array_sp_crossover(self.biases_hidden1_hidden2, s_b2)

        c1_w3, c2_w3 = self.__2d_array_sp_crossover(self.weights_hidden2_output, s_w3)
        c1_b3, c2_b3 = self.__1d_array_sp_crossover(self.biases_hidden2_output, s_b3)

        child1 = SnakeBrain(weights=[c1_w1, c1_w2, c1_w3], biases=[c1_b1, c1_b2, c1_b3])
        child2 = SnakeBrain(weights=[c2_w1, c2_w2, c2_w3], biases=[c2_b1, c2_b2, c2_b3])

        return child1, child2

    @staticmethod
    def __2d_array_sp_crossover(arr1, arr2):
        """
        A helper function that crosses over the rows of two arrays at random points
        :param arr1:
        :param arr2:
        :return: a crossover array comprising of the previous two arrays
        """
        tmp = arr1.copy()
        for i, row in enumerate(arr1):
            c_p = random.randint(0, len(row))
            arr1[i][c_p:] = arr2[i][c_p:]
            arr2[i][c_p:] = tmp[i][c_p:]

        return arr1, arr2

    @staticmethod
    def __1d_array_sp_crossover(arr1, arr2):
        c_p = random.randint(0, len(arr1))
        tmp = arr1.copy()
        arr1[c_p:] = arr2[c_p:]
        arr2[c_p:] = tmp[c_p:]
        return arr1, arr2

    def uniform_crossover(self, second_brain):
        """
        Crossover two brains, using uniform co, creating a child with a random amount of each it's parents qualities
        Uniform crossover uses a fixed mixing ratio between two parents, in this case 0.5, each elements has a 0.5
        chance of being from one parent or the other.

        :param second_brain: the brain to crossover with
        :return: SnakeBrain children of this brain and the second_brain
        """

        # The weights and biases for the second snake
        s_w1 = second_brain.weights_input_hidden1
        s_b1 = second_brain.biases_input_hidden1

        s_w2 = second_brain.weights_hidden1_hidden2
        s_b2 = second_brain.biases_hidden1_hidden2

        s_w3 = second_brain.weights_hidden2_output
        s_b3 = second_brain.biases_hidden2_output

        # Childs weights and biases

        c1_w1, c2_w1 = self.__2d_array_un_crossover(self.weights_input_hidden1, s_w1)
        c1_b1, c2_b1 = self.__1d_array_un_crossover(self.biases_input_hidden1, s_b1)

        c1_w2, c2_w2 = self.__2d_array_un_crossover(self.weights_hidden1_hidden2, s_w2)
        c1_b2, c2_b2 = self.__1d_array_un_crossover(self.biases_hidden1_hidden2, s_b2)

        c1_w3, c2_w3 = self.__2d_array_un_crossover(self.weights_hidden2_output, s_w3)
        c1_b3, c2_b3 = self.__1d_array_un_crossover(self.biases_hidden2_output, s_b3)

        child1 = SnakeBrain(weights=[c1_w1, c1_w2, c1_w3], biases=[c1_b1, c1_b2, c1_b3])
        child2 = SnakeBrain(weights=[c2_w1, c2_w2, c2_w3], biases=[c2_b1, c2_b2, c2_b3])

        return child1, child2

    @staticmethod
    def __2d_array_un_crossover(arr1, arr2):
        """
        Helper function for performing uniform crossover on 2d arrays
        :param arr1:
        :param arr2:
        :return: the two child arrays
        """
        parents = [arr1, arr2]
        child1 = []
        child2 = []

        for i, row in enumerate(arr1):
            tmp_child1 = []
            tmp_child2 = []
            for n, _ in enumerate(row):
                p = random.randint(0, 1)
                tmp_child1.append(parents[p][i][n])
                tmp_child2.append(parents[abs(p-1)][i][n])
            child1.append(tmp_child1)
            child2.append(tmp_child2)

        return np.array(child1), np.array(child2)

    @staticmethod
    def __1d_array_un_crossover(arr1, arr2):
        """
        Helper function for performing uniform crossover on 1d arrays
        :param arr1:
        :param arr2:
        :return: the two child arrays
        """

        parents = [arr1, arr2]
        child1 = []
        child2 = []

        for n, _ in enumerate(arr1):
            p = random.randint(0, 1)
            child1.append(parents[p][n])
            child2.append(parents[abs(p - 1)][n])
        return np.array(child1), np.array(child2)

    def mutate(self):
        """
        Mutate weights and biases, 2 mutations per row of the arrays by default
        :return:
        """
        self.weights_input_hidden1 = self.__2d_array_mutate(self.weights_input_hidden1)
        self.biases_input_hidden1 = self.__1d_array_mutate(self.biases_input_hidden1)

        self.weights_hidden1_hidden2 = self.__2d_array_mutate(self.weights_hidden1_hidden2)
        self.biases_hidden1_hidden2 = self.__1d_array_mutate(self.biases_hidden1_hidden2)

        self.weights_hidden2_output = self.__2d_array_mutate(self.weights_hidden2_output)
        self.biases_hidden2_output = self.__1d_array_mutate(self.biases_hidden2_output)

    @staticmethod
    def __2d_array_mutate(arr1):
        """
        A helper function that mutates
        :return: an array comprising of arr1 but with two random values per row mutated
        """

        for i, row in enumerate(arr1):
            i_1 = random.randint(0, len(row) - 1)
            i_2 = random.randint(0, len(row) - 1)
            arr1[i][i_1] = random.uniform(-1, 1)
            arr1[i][i_2] = random.uniform(-1, 1)

        return arr1

    @staticmethod
    def __1d_array_mutate(arr1,):
        """
        A helper function that mutates
        :return: an array comprising of arr1 but with two random values per row mutated
        """

        i_1 = random.randint(0, len(arr1) - 1)
        i_2 = random.randint(0, len(arr1) - 1)
        arr1[i_1] = random.uniform(-1, 1)
        arr1[i_2] = random.uniform(-1, 1)

        return arr1

    def play(self, graphical=True, delay: float=0., fruits=True):
        """
        Instruct this brain to play a game of snake.

        :param graphical: Whether or not to show the game, boosts performance by 70% without it
        :param delay: Optional delay between steps, easier to see what the snake is doing
        :param fruits: Whether or not to spawn fruits
        :return: Score, age
        """
        # The game of 'snake!' that this brain will use
        self.game = snake_custom.SnakeGame(graphical, fruits)

        while True:
            # Get what the snake can 'see'
            input_from_game = self.game.snake.look()

            # Feed forward the values from the snake's vision
            output_array = self.get_output(np.array(input_from_game))
            # Get the index of the highest value from the output_array
            # index: 0 => North, 1 => East, 2 => South, 3 => West
            output = np.argmax(output_array)

            # Returns false on snake surviving a game step, score and age on it dying
            result = self.game.step(output)

            # Game has ended
            if type(result) == tuple:
                # Retrun score, age
                return result[0], result[1]

            if delay:
                sleep(delay)


if __name__ == '__main__':
    # for testing
    while True:
        testnet = SnakeBrain()
        print(testnet.play(True, 0.1))
