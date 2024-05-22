import copy

from helper.config import Config
from helper.random import Random
from operators.crossover.abstract_crossover import AbstractCrossover


class SinglePointCrossover(AbstractCrossover):
    def crossover(self, individual_1, individual_2):
        conf = Config()
        rand = Random()


        ind_1 = copy.deepcopy(individual_1)
        ind_2 = copy.deepcopy(individual_2)
        w_dish_types = rand.random_int(conf.DISH_TYPE_SIZE)
        w_days_1 = rand.random_int(conf.NUMBER_OF_DAYS)
        w_days_2 = rand.random_int(conf.NUMBER_OF_DAYS)
        row_1 = ind_1.days[w_days_1].dish_types.iloc[w_dish_types, :]
        row_2 = ind_2.days[w_days_2].dish_types.iloc[w_dish_types, :]
        ind_2.days[w_days_2].dish_types.iloc[w_dish_types, :] = row_1
        ind_1.days[w_days_1].dish_types.iloc[w_dish_types, :] = row_2
        return ind_1, ind_2
