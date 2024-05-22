import numpy as np
from past.builtins import xrange

import constants
from fitnesses.abstract_fitness import AbstractFitness


class MealGroupFitness(AbstractFitness):
    """
    MealGroupFitness is a fitness function that compares repetition of the meal groups.
    """

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """

        ind = [day.dish_types._get_column_array(constants.MEAL_GROUP_INDEX) for day in individual.days]
        all_values = np.concatenate([x for x in ind])
        total_fitness = 0
        prev = all_values[0:3]
        for i in xrange(3, len(all_values), 3):
            # get the next three items from my_list
            curr = all_values[i:i + 3]
            two_day = np.concatenate([prev, curr])
            total_fitness += np.unique(two_day).__len__() / two_day.__len__()
            prev = curr

        return 1 - (total_fitness / (individual.days.__len__() - 1))

    def get_name(self):
        return "MealGroupFitness"

    def get_description(self):
        return "Compare the repetition of meal groups."
