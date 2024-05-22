from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas import DataFrame

import constants
from fitnesses.abstract_fitness import AbstractFitness


@dataclass
class FitnessFunctions:
    function: AbstractFitness = field(default_factory=AbstractFitness)
    weight: float = field(default=1)
    value: float = field(default=0.0)
    is_calculated: bool = field(default=False)


@dataclass
class Day:
    dish_types: np.array = field(default_factory=np.array)


@dataclass
class Solution:
    days: list = field(default_factory=list)
    fitness_functions: list = field(default_factory=list)
    total_fitness: float = 0.0
