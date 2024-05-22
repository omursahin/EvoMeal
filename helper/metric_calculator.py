import numpy as np
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

from helper.config import Config


class MetricCalculator:
    def __init__(self):
        conf = Config()
        self.ref_point = np.array([1.2] * conf.FITNESS_FUNCTIONS.__len__())
        self.pareto_front = np.array([0] * conf.FITNESS_FUNCTIONS.__len__())

    def calculate_hypervolume(self, res):
        ind = HV(ref_point=self.ref_point)
        return ind(res.F)

    def calculate_gd(self, res):
        # calculate the generational distance
        ind = GD(self.pareto_front)
        return ind(res.F)

    def calculate_gd_p(self, res):
        # calculate the generational distance plus
        ind = GDPlus(self.pareto_front)
        return ind(res.F)

    def calculate_igd(self, res):
        # calculate the inverted generational distance
        ind = IGD(self.pareto_front)
        return ind(res.F)

    def calculate_igd_p(self, res):
        # calculate the inverted generational distance plus
        ind = IGDPlus(self.pareto_front)
        return ind(res.F)
