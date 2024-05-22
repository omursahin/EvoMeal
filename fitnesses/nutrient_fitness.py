import numpy as np

import constants
from fitnesses.abstract_fitness import AbstractFitness


class NutrientFitness(AbstractFitness):
    """
    NutrientFitness is a fitness function that compares the nutrient levels of the individual to the nutrient levels of the target.
    """

    def calculate_difference(self, current, target):
        tolerance = 0.33
        sum_of_ing = np.sum(current)
        calc_abs = np.abs(target - sum_of_ing)
        percent_diff = calc_abs / target
        if (1 - tolerance) * target <= sum_of_ing <= (1 + tolerance) * target:
            return 1
        else:
            return 1 - percent_diff

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """
        from helper.config import Config
        conf = Config()
        sum = 0
        for days in individual.days:
            energy = days.dish_types._get_column_array(constants.ENERGY_INDEX)
            cho = days.dish_types._get_column_array(constants.CHO_INDEX)
            protein = days.dish_types._get_column_array(constants.PROTEIN_INDEX)
            fat = days.dish_types._get_column_array(constants.FAT_INDEX)

            calc_energy = self.calculate_difference(energy, conf.ENERGY)
            calc_cho = self.calculate_difference(cho, conf.CHO)
            calc_protein = self.calculate_difference(protein, conf.PROTEIN)
            calc_fat = self.calculate_difference(fat, conf.FAT)

            sum += (calc_energy + calc_cho + calc_protein + calc_fat) / 4
        return 1 - sum / individual.days.__len__()

    def get_name(self):
        return "NutrientFitness"

    def get_description(self):
        return "Compare the nutrients individual to the nutrients of the target."
