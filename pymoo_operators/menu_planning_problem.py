import string
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem

import constants
from fitnesses.color_fitness import ColorFitness
from fitnesses.consistency_fitness import ConsistencyFitness
from fitnesses.main_ingredient_fitness import MainIngredientFitness
from fitnesses.nutrient_fitness import NutrientFitness
from helper.config import Config
from solution import FitnessFunctions


class MenuPlanningProblem(ElementwiseProblem):

    def __init__(self, argv, file_path='./data/dataset.csv', df=None):
        self.conf = Config(argv)
        super().__init__(n_var=1, n_obj=self.conf.FITNESS_FUNCTIONS.__len__())

        try:
            if df is not None:
                self.df = df
            else:
                self.df = pd.read_csv(file_path)
        except:
            print("File not found")
            return

        self.first_dish_type = self.df[self.df['dish_type'] == 1]
        self.second_dish_type = self.df[self.df['dish_type'] == 2]
        self.third_dish_type = self.df[self.df['dish_type'] == 3]

        self.fitness_functions = list()
        for ff in self.conf.FITNESS_FUNCTIONS:
            self.fitness_functions.append(FitnessFunctions(function=ff['function'], weight=ff['weight']))

    def get_one_dish_type(self, dish_type):
        return self.df[self.df._get_column_array(constants.DISH_TYPE_INDEX) == dish_type + 1].sample()

    def _evaluate(self, x, out, *args, **kwargs):
        fitness_list = []
        for ff in x[0].fitness_functions:
            ff.value = ff.function.fitness(x[0])
            ff.is_calculated = True
            fitness_list.append(ff.value)
        x[0].total_fitness = sum(fitness_list) / len(fitness_list)

        out["F"] = np.array(fitness_list, dtype=float)
