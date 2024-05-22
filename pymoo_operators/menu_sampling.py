import copy

import numpy as np
from pandas import DataFrame
from pymoo.core.sampling import Sampling

from solution import Solution, Day


class MenuSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            X[i, 0] = self.generate_solution(problem)

        return X

    def generate_solution(self, problem):
        days = list()
        for i in range(5):
            dish_types = DataFrame()
            dish_types = dish_types.append(problem.first_dish_type.sample())
            dish_types = dish_types.append(problem.second_dish_type.sample())
            dish_types = dish_types.append(problem.third_dish_type.sample())
            days.append(Day(dish_types))
        solution = Solution(days, fitness_functions=copy.deepcopy(problem.fitness_functions))
        return solution
