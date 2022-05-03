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

    def model_input_data(self, algo):

        self.WS = f"{self.name}_ws"

        idata = super().model_input_data(algo)
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

    def calculate(self, algo, mdata, fdata, pdata):

        pdata[FV.WS][:] = mdata[self.WS][:, None]

        if self.wd is not None:
            pdata[FV.WD][:] = self.wd
        if self.ti is not None:
            pdata[FV.TI][:] = self.ti
        if self.rho is not None:
            pdata[FV.RHO][:] = self.rho
        
        return {v: pdata[v] for v in self.output_point_vars(algo)}
