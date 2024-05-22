import copy

import numpy as np

from helper.random import Random
from operators.mutation.abstract_mutation import AbstractMutation


class Mutation(AbstractMutation):
    def mutate(self, individual, get_one_dish_type):
        from helper.config import Config
        conf = Config()
        rand = Random()
        ind = copy.deepcopy(individual)
        w_dish_types = rand.random_int(conf.DISH_TYPE_SIZE)
        w_days_1 = rand.random_int(conf.NUMBER_OF_DAYS)
        new_row = get_one_dish_type(w_dish_types)
        ind.days[w_days_1].dish_types.iloc[[w_dish_types], :] = new_row
        return ind
