import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC

class SetXYHD(TurbineModel):

    def __init__(self, set_XY=True, set_H=True, set_D=True):
        super().__init__()

        self.set_XY = set_XY
        self.set_H  = set_H
        self.set_D  = set_D
    
    def output_farm_vars(self, algo):
        ovars = []
        if self.set_XY:
            ovars.append(FV.X)
            ovars.append(FV.Y)
        if self.set_H:
            ovars.append(FV.H)
        if self.set_D:
            ovars.append(FV.D)
        return ovars
    
    def calculate(self, algo, fdata, st_sel):

        n_states   = len(fdata[FV.STATE])
        n_turbines = algo.n_turbines

        out = {}
        if self.set_XY or self.set_H:
            fdata[FV.TXYH] = np.zeros((n_states, n_turbines, 3), dtype=FC.DTYPE)
            if self.set_XY:
                out[FV.X] = fdata[FV.TXYH][..., 0] 
                out[FV.Y] = fdata[FV.TXYH][..., 1] 
            if self.set_H:
                out[FV.H] = fdata[FV.TXYH][..., 2] 
        if self.set_D:
            out[FV.D] = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)

        for ti in range(n_turbines):
            ssel = st_sel[:, ti]
            if np.any(ssel):

                if np.all(ssel):
                    ssel = np.s_[:]
                    
                if self.set_XY:
                    out[FV.X][ssel, ti] = algo.farm.turbines[ti].xy[0]
                    out[FV.Y][ssel, ti] = algo.farm.turbines[ti].xy[1]

                if self.set_H:
                    H = algo.farm.turbines[ti].H
                    if H is None:
                        H = algo.farm_controller.turbine_types[ti].H
                    out[FV.H][ssel, ti] = H
                            
                if self.set_D:
                    D = algo.farm.turbines[ti].D
                    if D is None:
                        D = algo.farm_controller.turbine_types[ti].D
                    out[FV.D][ssel, ti] = D    

        return out    
