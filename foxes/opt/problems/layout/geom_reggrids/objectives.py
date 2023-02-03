import numpy as np
from iwopy import Objective

class MaxN(Objective):

    def __init__(self, problem, name="maxN"):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float())

    def n_components(self):
        return 1

    def maximize(self):
        return [True]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return np.sum(valid)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return np.sum(valid, axis=1)[:, None]

class MinN(Objective):

    def __init__(self, problem, name="minN"):
        super().__init__(problem, name)

    def maximize(self):
        return [False]