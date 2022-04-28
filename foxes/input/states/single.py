import numpy as np

from foxes.core import States
import foxes.variables as FV
import foxes.constants as FC

class SingleStateStates(States):

    def __init__(
        self,
        ws,
        wd,
        ti=None,
        rho=None
    ):
        super().__init__()
        self.ws  = ws
        self.wd  = wd
        self.ti  = ti
        self.rho = rho
    
    def size(self):
        return 1

    def output_point_vars(self, algo):
        out = []
        if self.ws is not None:
            out.append(FV.WS)
        if self.wd is not None:
            out.append(FV.WD)
        if self.ti is not None:
            out.append(FV.TI)
        if self.rho is not None:
            out.append(FV.RHO)
        return out

    def weights(self, algo):
        return np.ones((1, algo.n_turbines), dtype=FC.DTYPE)

    def calculate(self, algo, fdata, pdata):

        n_points = pdata.n_points
        out      = {}

        if self.ws is not None:
            out[FV.WS] = np.full((1, n_points), self.ws, dtype=FC.DTYPE)
        if self.wd is not None:
            out[FV.WD] = np.full((1, n_points), self.wd, dtype=FC.DTYPE)
        if self.ti is not None:
            out[FV.TI] = np.full((1, n_points), self.ti, dtype=FC.DTYPE)
        if self.rho is not None:
            out[FV.RHO] = np.full((1, n_points), self.rho, dtype=FC.DTYPE)
        
        return out
