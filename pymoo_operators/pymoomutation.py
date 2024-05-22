import numpy as np
from pymoo.core.mutation import Mutation

from helper.random import Random


class PymooMutation(Mutation):
    def __init__(self):
        super().__init__()


    def _do(self, problem, X, **kwargs):
        rand = Random()
        # for each individual
        for i in range(len(X)):

            r = rand.random_float()
            if r < 0.4:
                X[i, 0] = problem.conf.OPERATORS.get('mutation').mutate(X[i, 0], problem.get_one_dish_type)

        return X
