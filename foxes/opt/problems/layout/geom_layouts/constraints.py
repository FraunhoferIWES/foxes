import numpy as np
from scipy.spatial.distance import cdist
from iwopy import Constraint

import foxes.constants as FC

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

class CMinDensity(Constraint):

    def __init__(self, problem, min_value, dfactor=1, name="min_density"):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float())
        self.min_value = min_value
        self.dfactor = dfactor

    def n_components(self):
        return 1
    
    def initialize(self, verbosity):
        super().initialize(verbosity)
        
        # define regular grid of probe points:
        geom = self.problem.boundary
        pmin = geom.p_min()
        pmax = geom.p_max()
        detlta = self.problem.min_dist / self.dfactor
        self._probes = np.stack(
            np.meshgrid(
                np.arange(pmin[0]-detlta, pmax[0]+2*detlta, detlta),
                np.arange(pmin[1]-detlta, pmax[1]+2*detlta, detlta),
                indexing='ij'
            ), axis=-1
        )
        nx, ny = self._probes.shape[:2]
        n = nx*ny
        self._probes = self._probes.reshape(n, 2)

        # reduce to points within geometry:
        valid = geom.points_inside(self._probes)
        self._probes = self._probes[valid]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        xy, valid = problem_results
        xy = xy[valid]
        dists = cdist(self._probes, xy)
        return np.nanmax(np.nanmin(dists, axis=1)) - self.min_value

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        n_pop = vars_float.shape[0]
        xy, valid = problem_results
        out = np.full(n_pop, 1e20, dtype=FC.DTYPE)
        for pi in range(n_pop):
            if np.any(valid[pi]):
                hxy = xy[pi][valid[pi]]
                dists = cdist(self._probes, hxy)
                out[pi] = np.nanmax(np.nanmin(dists, axis=1)) -  self.min_value
        return out[:, None]