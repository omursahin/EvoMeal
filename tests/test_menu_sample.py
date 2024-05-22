import unittest

from pymoo_operators.menu_planning_problem import MenuPlanningProblem
from pymoo_operators.menu_sampling import MenuSampling


class TestMenuSample(unittest.TestCase):

    def setUp(self):
        self.problem = MenuPlanningProblem(argv=None, file_path='../data/dataset.csv')
        self.sample_menu = MenuSampling()

    def test_sample_menu(self):
        self.assertTrue(self.sample_menu is not None)
        menu = self.sample_menu.generate_solution(self.problem)
        self.assertTrue(menu is not None)
        self.assertTrue(menu.total_fitness == 0)
        self.assertTrue(menu.days.__len__() == self.problem.conf.NUMBER_OF_DAYS)
        self.assertEquals(menu.fitness_functions.__len__(), self.problem.conf.FITNESS_FUNCTIONS.__len__())

