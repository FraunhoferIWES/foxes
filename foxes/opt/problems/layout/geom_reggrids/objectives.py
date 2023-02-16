import numpy as np
from iwopy import Objective
from scipy.spatial.distance import cdist

import foxes.constants as FC

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
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float())
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

class MaxGridSpacing(Objective):

    def __init__(self, problem, name="max_dxdy"):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float())

    def n_components(self):
        return 1

    def maximize(self):
        return [True]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        vflt = vars_float.reshape(self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, 2], vflt[:, 3])
        return np.min(delta)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        n_pop = vars_float.shape[0]
        vflt = vars_float.reshape(n_pop, self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, :, 2], vflt[:, :, 3])
        return np.min(delta, axis=1)[:, None]

class MaxDensity(Objective):

    def __init__(self, problem, dfactor=1, name="max_density"):
        super().__init__(problem, name, vnames_int=problem.var_names_int(), 
            vnames_float=problem.var_names_float())
        self.dfactor = dfactor

    def n_components(self):
        return 1

    def maximize(self):
        return [False]
    
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
        return np.max(np.min(dists, axis=1))

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        n_pop = vars_float.shape[0]
        xy, valid = problem_results
        out = np.full(n_pop, 1e20, dtype=FC.DTYPE)
        for pi in range(n_pop):
            if np.any(valid[pi]):
                hxy = xy[pi][valid[pi]]
                dists = cdist(self._probes, hxy)
                out[pi] = np.max(np.min(dists, axis=1))
        return out[:, None]
