import numpy as np

import constants
from fitnesses.abstract_fitness import AbstractFitness


class MainIngredientFitness(AbstractFitness):
    """
    MainIngredientFitness is a fitness function that compares the main ingredients of the individual to the main ingredients of the target.
    """

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param conf: The configuration of the simulation.
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """
        # IF day is 1, then the fitness calculated with different main ingredients / total main ingredients
        # IF day is else, then the fitness calculated with different(previous day's main ingredients + main ingredients) / total main ingredients for two days
        from helper.config import Config
        conf = Config()
        sum = 0
        previous_day = None
        for idx, days in enumerate(individual.days):
            if idx == 0:
                main_ing_1 = days.dish_types._get_column_array(constants.MAIN_INGREDIENTS_INDEX)
                main_ing_2 = days.dish_types._get_column_array(constants.MAIN_INGREDIENTS_2_INDEX)
                merged_array = np.concatenate((main_ing_1, main_ing_2))
                filtered_ings = merged_array[~np.isnan(merged_array)]
                values, counts = np.unique(filtered_ings, return_counts=True)
                sum += counts.__len__() / filtered_ings.__len__()

                previous_day = merged_array
            else:
                main_ing_1 = days.dish_types._get_column_array(constants.MAIN_INGREDIENTS_INDEX)
                main_ing_2 = days.dish_types._get_column_array(constants.MAIN_INGREDIENTS_2_INDEX)
                merged_array = np.concatenate((main_ing_1, main_ing_2))
                merged_array_prev = np.concatenate((main_ing_1, main_ing_2, previous_day))
                filtered_ings = merged_array_prev[~np.isnan(merged_array_prev)]
                values, counts = np.unique(filtered_ings, return_counts=True)
                sum += counts.__len__() / filtered_ings.__len__()

                previous_day = merged_array

        return 1 - (sum / conf.NUMBER_OF_DAYS)

    def get_name(self):
        return "MainIngredientFitness"

    def get_description(self):
        return "Compare the main ingredients individual to the nutrients of the target."
