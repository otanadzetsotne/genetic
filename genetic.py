import random
import pickle
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from dtypes import *


class Genetic:
    __best_individual_path = 'best.pickle'

    __data_train = None
    __data_validation = None
    __data_test = None
    __labels_train = None
    __labels_validation = None
    __labels_test = None

    def __init__(
            self,
            input_shape: int,
            output_shape: int,
            data: np.ndarray,
            labels: np.ndarray,
            genome_len_max: int = 3,
            genome_len_min: int = 1,
            genome_width_max: int = 100,
            genome_width_min: int = 10,
            model_activations: list = activations,
            is_auto_encoder: bool = False,
    ):
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.__genome_len_max = genome_len_max
        self.__genome_len_min = genome_len_min
        self.__genome_width_max = genome_width_max
        self.__genome_width_min = genome_width_min
        self.__model_activations = model_activations
        self.__is_auto_encoder = is_auto_encoder

        # prepare data splitting
        data_len = len(data)
        counter_train = int((data_len * .65) // 1)
        counter_validation = int((data_len * .85) // 1)
        # save data
        self.__class__.__data_train = data[:counter_train]
        self.__class__.__data_validation = data[counter_train:counter_validation]
        self.__class__.__data_test = data[counter_validation:]
        # save labels
        self.__class__.__labels_train = labels[:counter_train]
        self.__class__.__labels_validation = labels[counter_train:counter_validation]
        self.__class__.__labels_test = labels[counter_validation:]

    def __random_shape(self) -> int:
        """
        Returns layer random shape
        :return: Layer output shape
        """

        return random.randint(self.__genome_width_min, self.__genome_width_max)

    def __random_activation(self) -> str:
        """
        Returns layer random activation
        :return: Layer activation function name
        """

        return random.choice(self.__model_activations)

    def __make_dna(
            self,
            layer_shape: Optional[int] = None,
            layer_activation: Optional[str] = None,
    ) -> DNA:
        """
        Creates genome's DNA
        :param layer_shape:
        :param layer_activation:
        :return: DNA code
        """

        layer_shape = layer_shape if layer_shape else self.__random_shape()
        layer_activation = layer_activation if layer_activation else self.__random_activation()

        return [layer_shape, layer_activation]

    def __make_individual(self) -> Individual:
        """
        Creates individual's genome
        :return: Individual
        """

        genome_len = random.randint(self.__genome_len_min, self.__genome_len_max)
        individual = [self.__make_dna() for _ in range(genome_len)]
        individual.append(self.__make_dna(self.__output_shape))

        return individual

    def __make_population(
            self,
            population_len: int
    ) -> Population:
        """
        Creates population of individuals
        :param population_len:
        :return: Population
        """

        return [self.__make_individual() for _ in range(population_len)]

    def __mutate(self, individual: Individual) -> Individual:
        """
        Create random DNA's mutation
        :param individual:
        :return: Mutated individual
        """

        genome_to_mutate = random.randint(0, len(individual) - 2)
        individual_mutated = individual[:genome_to_mutate] + [self.__make_dna()] + individual[genome_to_mutate + 1:]

        return individual_mutated

    @staticmethod
    def __mating(population) -> Parents:
        """
        Creates couples of individuals
        :param population:
        :return: Parents
        """

        couples = []
        # creating couples from closer individuals
        for k, individual in enumerate(population[:-1]):
            couples.append([
                population[k],
                population[k + 1]
            ])
        # create best/worst couple for strong genome
        couples.append([population[0], population[-1]])

        return couples

    @staticmethod
    def __pairing(parents: Parents) -> Population:
        """
        Pairs parents for born children
        :param parents:
        :return: Children
        """

        def pair(p_0: Individual, p_1: Individual):
            """
            Pairs parents genome to create childes
            :param p_0:
            :param p_1:
            :return: Child
            """

            genome_cut_index = random.randint(1, min(len(p_0), len(p_1)) - 1)
            child = p_0[:genome_cut_index] + p_1[genome_cut_index:]
            return child

        return [pair(parent_0, parent_1) for parent_0, parent_1 in parents]

    def __make_model(self, individual: Individual):
        """
        Create keras model from individual's genome
        :param individual:
        :return: Keras model
        """

        def add_layers(genome, base_structure):
            """
            Add layers to model structure
            :param genome:
            :param base_structure:
            :return: Keras model structure
            """

            for dna in genome:
                shape = dna[0]
                activation = dna[1]
                base_structure = layers.Dense(shape, activation=activation)(base_structure)

            return base_structure

        # create model layers
        input_layer = tf.keras.Input(shape=(self.__input_shape,))
        structure = add_layers(individual, input_layer)
        structure = add_layers(individual[1::-1], structure) if self.__is_auto_encoder else structure
        # create model
        model = tf.keras.Model(input_layer, structure)
        model.compile(
            optimizer='Adam',
            loss='mean_squared_error',
        )

        return model

    def __fit(
            self,
            population: list,
    ) -> Scores:
        """
        Train models by population genomes and get scores
        :param population:
        :return: Scores
        """

        scores = []
        # make score for each individual in population
        for individual in population:
            # create model by individuals genome
            model = self.__make_model(individual)
            # train model
            model.fit(
                self.__class__.__data_train,
                self.__class__.__labels_train,
                epochs=30,
                batch_size=min(256, len(self.__class__.__data_train)),
                validation_data=(self.__class__.__data_validation, self.__class__.__labels_validation),
                use_multiprocessing=True,
                verbose=False,
            )
            # get loss score
            score = model.evaluate(self.__class__.__data_test, self.__class__.__labels_test)
            score = float(score)
            scores.append(score)

        return scores

    @staticmethod
    def __random_die(
            population: Population,
            dies: int
    ) -> Population:
        """
        Imitates random dying
        :param population:
        :param dies:
        :return: Population
        """

        [population.pop(random.randint(0, len(population) - 1)) for _ in range(dies)]
        return population

    def evolution(
            self,
            population_len: int = 10,
    ):
        """
        Initialize evolution object with environment variables
        :param population_len: quantity of individuals in population
        """

        best_score = 0
        population = self.__make_population(population_len)
        print(f'Population created: {population}')

        while True:
            # get population scores
            scores = self.__fit(population)
            print(f'Got scores: {scores}')
            print(f'Best score: {min(scores)}')

            # sort population by scores
            population = [individual for _, individual in sorted(zip(scores, population))]
            print(f'Population sorted: {population}')

            # save best individual
            individual_best = population[0]
            print(f'Best individual: {individual_best}')
            individual_best_mutated = self.__mutate(individual_best)
            print(f'Best individual mutated: {individual_best_mutated}')

            # check best individual's progression
            best_score_tmp = max(scores)
            if best_score_tmp > best_score:
                best_score = best_score_tmp
                with open(self.__best_individual_path, 'wb') as f:
                    pickle.dump(individual_best, f)
                    print(f'Best individual saved in pickle dump: {self.__best_individual_path}')

            # evolution process
            couples = self.__mating(population)
            print(f'Couples created: {couples}')
            children = self.__pairing(couples)
            print(f'Children created: {children}')
            children = self.__random_die(children, 3)
            print(f'Children died randomly: {children}')

            # create new population for next epoch
            population = [individual_best] + [individual_best_mutated] + children + [self.__make_individual()]
            print(f'New population created: {population}')
