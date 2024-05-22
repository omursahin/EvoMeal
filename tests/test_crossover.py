import unittest

from operators.crossover.single_point_crossover import SinglePointCrossover


class TestCrossover(unittest.TestCase):

    def test_crossover(self):
        single_point_crossover = SinglePointCrossover()
        self.assertTrue(single_point_crossover is not None)
