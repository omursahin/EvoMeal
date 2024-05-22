import datetime
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.indicators.kktpm import KKTPM
from pymoo.visualization.scatter import Scatter

import constants
from helper import recorder
from helper.metric_calculator import MetricCalculator
from solution import Solution


class Reporter:
    def __init__(self, config):
        self.pymoo_header = "algorithm, run, iteration, experiment_id, number_of_population, number_of_generation, current_generation, " \
                            "total_fitness, hypervolume, gd, gd_p, igd, igd_p, energy, cho, protein, fat, time"
        self.header = "run, iteration, experiment_id, number_of_population, number_of_generation, current_generation, " \
                      "total_fitness"
        self.csv_text = "{}, {}, {}, {}, {}, {}, {}, {}"
        self.pymoo_csv_text = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}"
        self.config = config

        for ff in config.FITNESS_FUNCTIONS:
            self.header += ", " + ff.get('function').get_name()
            self.pymoo_header += ", " + ff.get('function').get_name()
            self.csv_text += ", {}"
            self.pymoo_csv_text += ", {}"

        self.header += "\n"
        self.pymoo_header += "\n"
        self.csv_text += "\n"
        self.pymoo_csv_text += "\n"

        experiment_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "").replace(":", "")
        self.experiment_id = experiment_id + "-" + config.EXPERIMENT_NAME
        self.file_path = "%s/%s/%s.csv" % (config.OUTPUTS_FOLDER_NAME, config.CSV_FOLDER_NAME, self.experiment_id)
        self.pymoo_file_path = "%s/%s/%s_pymoo.csv" % (
            config.OUTPUTS_FOLDER_NAME, config.CSV_FOLDER_NAME, self.experiment_id)
        self.mat_file_path = "%s/%s" % (config.OUTPUTS_FOLDER_NAME, self.experiment_id)

        self.html_file_path = "%s/%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.HTML_FOLDER_NAME, self.experiment_id)
        self.fig_file_path = "%s/%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.hyp_file_path = "%s/%s/%s_hypervolume" % (
            config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.gd_file_path = "%s/%s/%s_gd_p" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.gd_p_file_path = "%s/%s/%s_gd_p" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.igd_file_path = "%s/%s/%s_igd" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.igd_p_file_path = "%s/%s/%s_igd_p" % (
            config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.conv_file_path = "%s/%s/%s_mean_conv" % (
            config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME, self.experiment_id)
        self.bg_colors = [
            'bg-info',
            'bg-success',
            'bg-warning',
            'bg-danger',
            'bg-primary',
            'bg-secondary',
            'bg-dark',
            'bg-light',
            'bg-white',
        ]
        if not os.path.exists(config.OUTPUTS_FOLDER_NAME):
            os.makedirs(config.OUTPUTS_FOLDER_NAME)

        if not os.path.exists("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.HTML_FOLDER_NAME)):
            os.makedirs("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.HTML_FOLDER_NAME))

        if not os.path.exists("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.CSV_FOLDER_NAME)):
            os.makedirs("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.CSV_FOLDER_NAME))
        if not os.path.exists("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME)):
            os.makedirs("%s/%s" % (config.OUTPUTS_FOLDER_NAME, config.FIG_FOLDER_NAME))

        self.html_string_2 = '''
            <html>
              <head><title>Menu</title>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>           </head>
              <body>
                {table}
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: {color_percentage}%" aria-valuenow="{color_percentage}" aria-valuemin="0" aria-valuemax="100">Color: {color}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" role="progressbar" style="width: {consistency_percentage}%" aria-valuenow="{consistency_percentage}" aria-valuemin="0" aria-valuemax="100">Consistency: {consistency}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: {main_ing_percentage}%" aria-valuenow="{main_ing_percentage}" aria-valuemin="0" aria-valuemax="100">Main Ingredients: {main_ing}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-danger" role="progressbar" style="width: {nutrients_percentage}%" aria-valuenow="{nutrients_percentage}" aria-valuemin="0" aria-valuemax="100">Nutrients: {nutrients}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-secondary" role="progressbar" style="width: {repetition_percentage}%" aria-valuenow="{repetition_percentage}" aria-valuemin="0" aria-valuemax="100">Repetition: {repetition}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-success" role="progressbar" style="width: {meal_group_percentage}%" aria-valuenow="{meal_group_percentage}" aria-valuemin="0" aria-valuemax="100">Meal Group: {meal_group}</div>
                </div>
                <div class="progress">
                  <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" role="progressbar" style="width: {footprint_percentage}%" aria-valuenow="{footprint_percentage}" aria-valuemin="0" aria-valuemax="100">Footprint: {footprint}</div>
                </div>
              </body>
            </html>.
            '''
        self.html_string = '''
            <html>
              <head><title>Menu</title>
                    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
                    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.12.9/dist/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
                    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>           </head>
              <body>
                {table}
                {progress_bars}
              </body>
            </html>.
            '''

    def write_header(self):
        with open(self.file_path, 'a') as saveRes:
            saveRes.write(self.header)
            saveRes.close()

    def write_pymoo_header(self):
        with open(self.pymoo_file_path, 'a') as saveRes:
            saveRes.write(self.pymoo_header)
            saveRes.close()

    def write_row(self, run_id, it, cur_gen, fitnesses, total_fitness):
        values = [i.value for i in fitnesses]
        csv_text = self.csv_text.format(run_id, it, self.experiment_id, self.config.NUMBER_OF_POPULATION,
                                        self.config.MAXIMUM_EVALUATION, cur_gen, total_fitness, *values)

        with open(self.file_path, 'a') as saveRes:
            saveRes.write(csv_text)
            saveRes.close()

    def write_pymoo_row(self, algorithm, run_id, it, cur_gen, fitnesses, total_fitness, hypervolume, gd, gd_p, igd,
                        igd_p, energy, cho, protein, fat, time):
        float_format = '{:.3f}'

        values = [i.value for i in fitnesses]
        csv_text = self.pymoo_csv_text.format(algorithm, run_id, it, self.experiment_id,
                                              self.config.NUMBER_OF_POPULATION,
                                              self.config.MAXIMUM_EVALUATION, cur_gen, total_fitness, hypervolume, gd,
                                              gd_p, igd, igd_p, float_format.format(energy), float_format.format(cho),
                                              float_format.format(protein), float_format.format(fat), time, *values)

        with open(self.pymoo_file_path, 'a') as saveRes:
            saveRes.write(csv_text)
            saveRes.close()

    def highlight_max(self, x, color):
        return np.where(x > np.nanmax(x.to_numpy()), f"color: {color};", None)

    def print_solution(self, solution):
        index = 1
        for day in solution.days:
            print("Day:" + str(index))
            print(day.dish_types)
            index += 1

    def save_solution(self, solution):
        if self.config.SAVE_RESULTS:
            np.save(self.mat_file_path, solution)

    def load_solution(self, file_name):
        return Solution(np.load(file_name))

    def is_in_range(self, x, color):
        values = pd.Series([float(i.strip('%')) for i in x])
        return np.where(np.logical_and(values >= 100 - self.config.TOLERANCE, values <= 100 + self.config.TOLERANCE),
                        f"color: {color};", f"color: red;", )

    def generate_progress_bars(self, solution):
        progress_string = '''
                            <div class="progress">
                              <div class="progress-bar progress-bar-striped progress-bar-animated {color}" role="progressbar" style="width: {percentage}%" aria-valuenow="{percentage}" aria-valuemin="0" aria-valuemax="100">{fitness_name}: {value}</div>
                            </div>
                        '''
        progress_bars = ''
        for idx, fitness in enumerate(solution.fitness_functions):
            progress_bars += progress_string.format(percentage=np.round((1 - fitness.value) * 100, 2),
                                                    fitness_name=fitness.function.get_name(),
                                                    value=np.round(1 - fitness.value, 2),
                                                    color=self.bg_colors[idx % len(self.bg_colors)])
        return progress_bars

    def report(self, solution, run, algorithm, write_to_html=True):
        header = ['Day', 'Menu', 'Energy', 'P_Energy', 'Cho', 'P_Cho', 'Protein', 'P_Protein', 'Fat', 'P_Fat']
        df = pd.DataFrame(columns=header)
        df.columns = header
        float_format = '{:.2f}'
        float_format_percent = '%{:.2f}'
        for idx, day in enumerate(solution.days):
            meal = day.dish_types._get_column_array(constants.FOOD_INDEX)
            energy = np.round(day.dish_types._get_column_array(constants.ENERGY_INDEX).sum(), 4)
            cho = np.round(day.dish_types._get_column_array(constants.CHO_INDEX).sum(), 4)
            protein = np.round(day.dish_types._get_column_array(constants.PROTEIN_INDEX).sum(), 4)
            fat = np.round(day.dish_types._get_column_array(constants.FAT_INDEX).sum(), 4)

            data = {'Day': idx + 1,
                    'Menu': '\\n'.join(meal),
                    'Energy': float_format.format(energy),
                    'P_Energy': float_format_percent.format((energy / self.config.ENERGY) * 100),
                    'Cho': float_format.format(cho),
                    'P_Cho': float_format_percent.format((cho / self.config.CHO) * 100),
                    'Protein': float_format.format(protein),
                    'P_Protein': float_format_percent.format((protein / self.config.PROTEIN) * 100),
                    'Fat': float_format.format(fat),
                    'P_Fat': float_format_percent.format((fat / self.config.FAT) * 100)}
            df = df.append(data, ignore_index=True)
        out = df.style.apply(self.is_in_range, color='green',
                             subset=["P_Energy", "P_Cho", "P_Protein", "P_Fat"]).set_table_attributes(
            'class="table table-striped text-center"')
        if write_to_html:
            with open("{}_{}_{}.html".format(self.html_file_path, run, algorithm), 'w') as f:
                f.write(self.html_string.format(table=out.to_html().replace("\\n", "<br>"),
                                                progress_bars=self.generate_progress_bars(solution)))
        else:
            return self.html_string.format(table=out.to_html().replace("\\n", "<br>"),
                                           progress_bars=self.generate_progress_bars(solution))

    def show_and_save_plot(self, problem, res, run, algorithm):
        plot = Scatter()
        plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, color="red")
        plot.tight_layout = True
        plot.save("{}_{}_{}.png".format(self.fig_file_path, algorithm, run))
        if self.config.SHOW_PLOT:
            plot.show()
        else:
            plot.__del__()

    def show_and_save_metric_plots(self, res, algorithm, run):
        from pymoo.indicators.igd_plus import IGDPlus

        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation

        for algo in res.history:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)
            # retrieve the optimum from the algorithm
            opt = algo.opt
            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])

        opt = np.array([np.mean(e.opt[0].F) for e in res.history])
        hypervolume_metric = Hypervolume(ref_point=[1] * self.config.FITNESS_FUNCTIONS.__len__())
        gd_metric = GD(pf=np.array([0] * self.config.FITNESS_FUNCTIONS.__len__()), zero_to_one=False)
        gd_p_metric = GDPlus(pf=np.array([0] * self.config.FITNESS_FUNCTIONS.__len__()), zero_to_one=False)
        igd_metric = IGD(pf=np.array([0] * self.config.FITNESS_FUNCTIONS.__len__()), zero_to_one=False)
        igd_p_metric = IGDPlus(pf=np.array([0] * self.config.FITNESS_FUNCTIONS.__len__()), zero_to_one=False)

        hypervolume = [hypervolume_metric.do(_F) for _F in hist_F]
        gd = [gd_metric.do(_F) for _F in hist_F]
        gd_p = [gd_p_metric.do(_F) for _F in hist_F]
        igd = [igd_metric.do(_F) for _F in hist_F]
        igd_p = [igd_p_metric.do(_F) for _F in hist_F]

        self.draw_metric_plot(algorithm, n_evals, hypervolume, "Avg. Hypervolume of Pop",
                              "Hypervolume", "{}_{}.png".format(self.hyp_file_path, run))
        self.draw_metric_plot(algorithm, n_evals, gd, "Avg. GD of Pop",
                              "GD", "{}_{}_{}.png".format(self.gd_file_path, algorithm, run))
        self.draw_metric_plot(algorithm, n_evals, gd_p, "Avg. GD+ of Pop",
                              "GD+", "{}_{}_{}.png".format(self.gd_p_file_path, algorithm, run))
        self.draw_metric_plot(algorithm, n_evals, igd, "Avg. IGD of Pop",
                              "IGD", "{}_{}_{}.png".format(self.igd_file_path, algorithm, run))
        self.draw_metric_plot(algorithm, n_evals, igd_p, "Avg. IGD+ of Pop",
                              "IGD+", "{}_{}_{}.png".format(self.igd_p_file_path, algorithm, run))
        self.conv_plot(algorithm, n_evals, opt, run)

    def draw_metric_plot(self, algorithm, n_evals, result, title, metric_name, save_path):
        plt.clf()
        plt.plot(n_evals, result, color='black', lw=0.7, label=title)
        plt.scatter(n_evals, result, facecolor="none", edgecolor='black', marker="p")
        plt.title("Convergence - {}".format(algorithm))
        plt.xlabel("Function Evaluations")
        plt.ylabel(metric_name)
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        if self.config.SHOW_PLOT:
            plt.show()
        else:
            plt.clf()

    def history_writer(self, run, res, algorithm, time):
        calculator = MetricCalculator()
        float_format = '{:.2f}'
        float_format_percent = '%{:.2f}'
        for h in res.history:
            result = h.result()
            ind_fitnesses = [x[0].total_fitness for x in result.X]
            best_sol = result.X[np.argmin(ind_fitnesses)]
            best_sol = best_sol[0]
            hyp = calculator.calculate_hypervolume(result)
            gd = calculator.calculate_gd(result)
            gd_p = calculator.calculate_gd_p(result)
            igd = calculator.calculate_igd(result)
            igd_p = calculator.calculate_igd_p(result)
            energy = 0
            cho = 0
            protein = 0
            fat = 0
            for idx, day in enumerate(best_sol.days):
                energy += day.dish_types._get_column_array(constants.ENERGY_INDEX).sum()
                cho += day.dish_types._get_column_array(constants.CHO_INDEX).sum()
                protein += day.dish_types._get_column_array(constants.PROTEIN_INDEX).sum()
                fat += day.dish_types._get_column_array(constants.FAT_INDEX).sum()
            energy = energy / best_sol.days.__len__()
            cho = cho / best_sol.days.__len__()
            protein = protein / best_sol.days.__len__()
            fat = fat / best_sol.days.__len__()

            self.write_pymoo_row(algorithm, run, h.n_iter, h.evaluator.n_eval, best_sol.fitness_functions,
                                 best_sol.total_fitness,
                                 hyp, gd, gd_p, igd, igd_p, energy, cho, protein, fat, time)

    def conv_plot(self, algorithm, n_evals, opt, run):
        plt.clf()
        plt.title("Convergence for {} algoritm".format(algorithm))
        plt.plot(n_evals, opt, "--")
        plt.yscale("log")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Mean objective value")
        plt.tight_layout()
        plt.savefig("{}_{}_{}.png".format(self.conv_file_path, algorithm, run), dpi=300)
        if self.config.SHOW_PLOT:
            plt.show()
        else:
            plt.clf()

    def generate_solution_json(self, solution):
        response_dict = []
        float_format = '{:.2f}'
        float_format_percent = '%{:.2f}'
        fitness_dict = []

        for fitness in solution.fitness_functions:
            fitness_dict.append({'name': fitness.function.get_name(),
                                 'value': np.round(1 - fitness.value, 2),
                                 'percentage': np.round((1 - fitness.value) * 100, 2)})

        for idx, day in enumerate(solution.days):
            meal = day.dish_types._get_column_array(constants.FOOD_INDEX)
            energy = np.round(day.dish_types._get_column_array(constants.ENERGY_INDEX).sum(), 4)
            cho = np.round(day.dish_types._get_column_array(constants.CHO_INDEX).sum(), 4)
            protein = np.round(day.dish_types._get_column_array(constants.PROTEIN_INDEX).sum(), 4)
            fat = np.round(day.dish_types._get_column_array(constants.FAT_INDEX).sum(), 4)
            data = {'day': idx + 1,
                    'menu': [str(m) for m in meal],
                    'energy': float_format.format(energy),
                    'p_energy': float_format_percent.format(energy / self.config.ENERGY * 100),
                    'cho': float_format.format(cho),
                    'p_cho': float_format_percent.format(cho / self.config.CHO * 100),
                    'protein': float_format.format(protein),
                    'p_protein': float_format_percent.format(protein / self.config.PROTEIN * 100),
                    'fat': float_format.format(fat),
                    'p_fat': float_format_percent.format(fat / self.config.FAT * 100),
                    }
            response_dict.append(data)
        return {
            'data': response_dict,
            'algorithm': self.config.ALGORITHM,
            'fitnesses': fitness_dict,
            'config': {
                'energy': self.config.ENERGY,
                'cho': self.config.CHO,
                'protein': self.config.PROTEIN,
                'fat': self.config.FAT,
                'tolerance': self.config.TOLERANCE,
                'operators': {
                    'crossover': self.config.OPERATORS['crossover'].__class__.__name__,
                    'mutation': self.config.OPERATORS['mutation'].__class__.__name__,
                    'selection': self.config.OPERATORS['selection'].__class__.__name__,
                },
                'number_of_population': self.config.NUMBER_OF_POPULATION,
                'maximum_evaluation': self.config.MAXIMUM_EVALUATION,
                'random_seed': self.config.RANDOM_SEED
            }
        }
