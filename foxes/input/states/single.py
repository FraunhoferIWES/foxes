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

    def calculate(self, algo, mdata, fdata, pdata):

        if self.ws is not None:
            pdata[FV.WS][:] = self.ws
        if self.wd is not None:
            pdata[FV.WD][:] = self.wd
        if self.ti is not None:
            pdata[FV.TI][:] = self.ti
        if self.rho is not None:
            pdata[FV.RHO][:] = self.rho
        