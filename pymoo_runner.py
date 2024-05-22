import sys

import numpy as np
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.evaluator import Evaluator
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

from helper import recorder
from helper.metric_calculator import MetricCalculator
from helper.random import Random
from helper.reporter import Reporter
from pymoo_operators.pymoocrossover import PymooCrossover
from pymoo_operators.duplicates import Duplicates
from pymoo_operators.menu_planning_problem import MenuPlanningProblem
from pymoo_operators.pymoomutation import PymooMutation
from pymoo_operators.menu_sampling import MenuSampling
import timeit


def get_algorithm(algorithm_name, number_of_population):
    evaluator = Evaluator(evaluate_values_of=["F", "G", "dF", "dG"])

    if algorithm_name == 'NSGA2':
        algorithm = NSGA2(pop_size=number_of_population,
                          sampling=MenuSampling(),
                          crossover=PymooCrossover(),
                          mutation=PymooMutation(),
                          eliminate_duplicates=Duplicates(),
                          evaluator=evaluator)
    elif algorithm_name == 'NSGA3':
        ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=number_of_population)
        algorithm = NSGA3(pop_size=number_of_population,
                          ref_dirs=ref_dirs,
                          sampling=MenuSampling(),
                          crossover=PymooCrossover(),
                          mutation=PymooMutation(),
                          eliminate_duplicates=Duplicates(),
                          evaluator=evaluator)
    elif algorithm_name == 'SMSEMOA':
        algorithm = SMSEMOA(pop_size=number_of_population,
                            sampling=MenuSampling(),
                            crossover=PymooCrossover(),
                            mutation=PymooMutation(),
                            eliminate_duplicates=Duplicates(),
                            evaluator=evaluator)
    elif algorithm_name == 'MOEAD':
        ref_dirs = get_reference_directions("das-dennis", 6, n_partitions=number_of_population)
        algorithm = MOEAD(ref_dirs,
                          n_neighbors=number_of_population,
                          sampling=MenuSampling(),
                          crossover=PymooCrossover(),
                          mutation=PymooMutation(),
                          evaluator=evaluator)
    elif algorithm_name == 'AGEMOEA':
        algorithm = AGEMOEA(pop_size=number_of_population,
                            sampling=MenuSampling(),
                            crossover=PymooCrossover(),
                            mutation=PymooMutation(),
                            eliminate_duplicates=Duplicates(),
                            evaluator=evaluator)
    return algorithm


def main(argv):
    problem = MenuPlanningProblem(argv)

    algorithm = get_algorithm(problem.conf.ALGORITHM, problem.conf.NUMBER_OF_POPULATION)

    reporter = Reporter(problem.conf)
    reporter.write_pymoo_header()
    rand = Random()
    for run in range(problem.conf.RUN_TIME):
        start = timeit.default_timer()
        res = minimize(problem,
                       algorithm,
                       ('n_evals', problem.conf.MAXIMUM_EVALUATION),
                       seed=rand.random.get_state()[1][0],
                       save_history=True,
                       verbose=True)
        stop = timeit.default_timer()

        ind_fitnesses = [x[0].total_fitness for x in res.X]
        best_sol = res.X[np.argmin(ind_fitnesses)]
        best_sol = best_sol[0]
        reporter.report(best_sol, run, algorithm.__class__.__name__)
        reporter.history_writer(run, res, algorithm.__class__.__name__, stop - start)
        reporter.show_and_save_plot(problem, res, run, algorithm.__class__.__name__)
        reporter.show_and_save_metric_plots(res, algorithm.__class__.__name__, run)
        recorder.record(res)


if __name__ == '__main__':
    main(sys.argv[1:])
