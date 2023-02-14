import numpy as np
from iwopy import Objective

class OMaxN(Objective):

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

class OMinN(OMaxN):

    def __init__(self, problem, name="ominN"):
        super().__init__(problem, name)

    def maximize(self):
        return [False]

class OFixN(Objective):

    def __init__(self, problem, N, name="ofixN"):
        super().__init__(problem, name)
        self.N = N

    def n_components(self):
        return 1

    def maximize(self):
        return [False]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        N = np.sum(valid, dtype=np.float64)
        return np.maximum(N - self.N, self.N - N)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        N = np.sum(valid, axis=1, dtype=np.float64)[:, None]
        return np.maximum(N - self.N, self.N - N)

