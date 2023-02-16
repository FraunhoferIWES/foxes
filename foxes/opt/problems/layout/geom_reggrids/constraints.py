import numpy as np
from iwopy import Constraint

class CMinN(Constraint):

    def __init__(self, problem, N, name="cminN", **kwargs):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float(), **kwargs)
        self.N = N

    def n_components(self):
        return 1

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return self.N - np.sum(valid)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return self.N - np.sum(valid, axis=1)[:, None]

class CMaxN(Constraint):

    def __init__(self, problem, N, name="cmaxN", **kwargs):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float(), **kwargs)
        self.N = N

    def n_components(self):
        return 1

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return np.sum(valid) - self.N

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        return np.sum(valid, axis=1)[:, None] - self.N

class CFixN(Constraint):

    def __init__(self, problem, N, name="cfixN", **kwargs):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float(), tol=0.1, **kwargs)
        self.N = N

    def n_components(self):
        return 2

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        vld = np.sum(valid)
        return np.array([self.N - vld, vld - self.N])

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        __, valid = problem_results
        vld = np.sum(valid, axis=1)
        return np.stack([self.N - vld, vld - self.N], axis=-1)
