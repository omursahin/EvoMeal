import numpy as np

import constants
from fitnesses.abstract_fitness import AbstractFitness


class RepetitionFitness(AbstractFitness):
    """
    RepetitionFitness is a fitness function that compares repetition of the menus.
    """

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """
        #ind = [x.dish_types['id'] for x in individual.days]
        #all_values = np.concatenate([x.values for x in ind])
        ind = [day.dish_types._get_column_array(constants.FOOD_ID) for day in individual.days]
        all_values = np.concatenate([x for x in ind])

        unique_values = np.unique(all_values)

        return 1 - (unique_values.__len__() / all_values.__len__())

    def get_name(self):
        return "RepetitionFitness"

    def get_description(self):
        return "Compare the repetition of menus."
