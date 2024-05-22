import numpy as np

import constants
from fitnesses.abstract_fitness import AbstractFitness


class ConsistencyFitness(AbstractFitness):
    """
    ConsistencyFitness is a fitness function that compares the consistency levels of the individual to the consistency levels of the target.
    """

    def fitness(self, individual):
        """
        Calculates the fitness of the individual.
        :param individual: The individual to calculate the fitness of.
        :return: The fitness of the individual.
        """
        # 0 -> Solid
        # 1 -> Liquid
        # 3 -> Not important
        # If 2 solid and 1 liquid, then fitness is 1
        # If 3 liquid, then fitness is 0
        # If 1 solid and 2 liquid, then fitness is 0.5
        # Not important will evaluated as best possible fitness
        sum = 0
        for days in individual.days:
            number_of_solid = 0
            number_of_liquid = 0
            number_of_not_important = 0

            consistency = days.dish_types._get_column_array(constants.CONSISTENCY_INDEX)
            filtered_consistency = consistency[~np.isnan(consistency)]
            values, counts = np.unique(filtered_consistency, return_counts=True)
            result = np.column_stack((values, counts))
            for i in result:
                if i[0] == 0:
                    number_of_solid = i[1]
                elif i[0] == 1:
                    number_of_liquid = i[1]
                elif i[0] == 3:
                    number_of_not_important = i[1]
            if number_of_not_important == 0:
                if number_of_solid == 3:
                    sum += 0
                if number_of_solid == 2 and number_of_liquid == 1:
                    sum += 1
                elif number_of_liquid == 3:
                    sum += 0
                elif number_of_solid == 1 and number_of_liquid == 2:
                    sum += 0.5
            else:
                if number_of_liquid == 2:
                    sum += 0.5
                else:
                    sum += 1

        return 1 - (sum / individual.days.__len__())

    def get_name(self):
        return "ConsistencyFitness"

    def get_description(self):
        return "Compare the consistency of individual to the consistency of the target."
