import numpy as np 

from foxes.core import RotorModel
import foxes.constants as FC

class GridRotor(RotorModel):
    """
    The regular grid rotor model, composed maximally
    of n x n points, possibly kicking out the outside points.

    Parameters
    ----------
    n: int
        The number of points along one direction,
        maximal number of points is N = n * n
    reduce: bool
        Flag for reduction to points actually representing
        an area with overlap with the circe, recalculating
        the self.weights accordingly
    nint: int
        Integration steps per element
    name: str, optional
        The model name

    Attributes
    ----------
    n: int
        The number of points along one direction,
        maximal number of points is N = n * n
    reduce: bool
        Flag for reduction to points actually representing
        an area with overlap with the circe, recalculating
        the self.weights accordingly
    nint: int
        Integration steps per element

    """

    def __init__(self, n, calc_vars, reduce=True, nint=200):
        super().__init__(calc_vars)

        self.n      = n
        self.reduce = reduce
        self.nint   = nint

    def __repr__(self):
        return super().__repr__() + f"(n={self.n})"
    
    def initialize(self, algo, farm_data):

        N     = self.n * self.n
        delta = 2. / self.n 
        x     = [ -1. + ( i + 0.5 ) * delta for i in range(self.n) ]
        x, y  = np.meshgrid(x, x, indexing='ij')

        self.dpoints = np.zeros([N, 3], dtype=FC.DTYPE)
        self.dpoints[:, 1] = x.reshape(N)
        self.dpoints[:, 2] = y.reshape(N)

        if self.reduce:
                    
            self.weights = np.zeros((self.n, self.n), dtype=FC.DTYPE)
            for i in range(0, self.n):
                for j in range(0, self.n):

                    d   = delta / self.nint 
                    hx  = [ x[i, j] - delta/2. + ( k + 0.5 ) * d for k in range(self.nint) ]
                    hy  = [ y[i, j] - delta/2. + ( k + 0.5 ) * d for k in range(self.nint) ]
                    pts = np.zeros((self.nint, self.nint, 2), dtype=FC.DTYPE)
                    pts[:, :, 0], pts[:, :, 1] = np.meshgrid(hx, hy, indexing='ij')
                    
                    d = np.linalg.norm(pts, axis=2)
                    self.weights[i, j] = np.sum(d <= 1.) / self.nint**2

            self.weights  = self.weights.reshape(N)
            sel           = self.weights > 0. 
            self.dpoints  = self.dpoints[sel]
            self.weights  = self.weights[sel]
            self.weights /= np.sum(self.weights)

        else:
            
            self.dpoints[:, 1] = x.reshape(N)
            self.dpoints[:, 2] = y.reshape(N)
            self.weights       = np.ones(N, dtype=FC.DTYPE) / N

        super().initialize(algo, farm_data)

    def n_rotor_points(self):
        return len(self.weights)

    def design_points(self):
        return self.dpoints
    
    def rotor_point_weights(self):
        return self.weights
