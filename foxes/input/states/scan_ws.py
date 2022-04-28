import numpy as np

from foxes.core import States
import foxes.variables as FV
import foxes.constants as FC

class ScanWS(States):

    def __init__(
        self,
        ws_list,
        wd,
        ti=None,
        rho=None
    ):
        super().__init__()

        self._wsl = np.array(ws_list)
        self.N    = len(ws_list)
        self.wd   = wd
        self.ti   = ti
        self.rho  = rho

    def input_farm_data(self, algo):

        self.WS = f"{self.name}_ws"

        idata = super().input_farm_data(algo)
        idata["data_vars"][self.WS] = ((FV.STATE, ), self._wsl)

        del self._wsl

        return idata

    def size(self):
        return self.N

    def output_point_vars(self, algo):
        pvars = [FV.WS]
        if self.wd is not None:
            pvars.append(FV.WD)
        if self.ti is not None:
            pvars.append(FV.TI)
        if self.rho is not None:
            pvars.append(FV.RHO)
        return pvars

    def weights(self, algo):
        return np.full((self.N, algo.n_turbines), 1./self.N, dtype=FC.DTYPE)

    def calculate(self, algo, fdata, pdata):

        n_states = fdata.n_states
        n_points = pdata.n_points
        
        out = {FV.WS: np.zeros((n_states, n_points), dtype=FC.DTYPE)}
        out[FV.WS][:] = fdata[self.WS][:, None]

        if self.wd is not None:
            out[FV.WD] = np.full((n_states, n_points), self.wd, dtype=FC.DTYPE)
        if self.ti is not None:
            out[FV.TI] = np.full((n_states, n_points), self.ti, dtype=FC.DTYPE)
        if self.rho is not None:
            out[FV.RHO] = np.full((n_states, n_points), self.rho, dtype=FC.DTYPE)

        return out
