import copy

import numpy as np

import constants
from helper.random import Random
from operators.selection.AbstractSelection import AbstractSelection


class RouletteSelection(AbstractSelection):
    def select(self, population):
        rand = Random()
        max = sum([i.total_fitness for i in population])
        selection_probs = [i.total_fitness / max for i in population]
        selection_indice = rand.random_choice(population.__len__(), selection_probs)
        return selection_indice, copy.deepcopy(population[selection_indice])
