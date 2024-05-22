import numpy as np

import constants
from fitnesses.abstract_fitness import AbstractFitness


class ColorFitness(AbstractFitness):
    """
    ColorFitness is a fitness function that compares the color of the individual to the color of the target.
    """

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param conf:
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """
        sum = 0
        for days in individual.days:
            colors = days.dish_types._get_column_array(constants.COLOR_INDEX)
            # if color is nan discarded to evaluate the fitness
            filtered_colors = colors[~np.isnan(colors)]
            if filtered_colors.__len__() == 0:
                sum += 1
            else:
                values, counts = np.unique(filtered_colors, return_counts=True)
                sum += counts.__len__() / filtered_colors.__len__()
        return 1 - (sum / individual.days.__len__())

    def get_name(self):
        return "ColorFitness"

    def get_description(self):
        return "Compare the color of the individual to the color of the target."
