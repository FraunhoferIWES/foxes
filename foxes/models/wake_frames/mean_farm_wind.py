import numpy as np

from foxes.core import WakeFrame
from foxes.tools import wd2uv
import foxes.variables as FV
import foxes.constants as FC

class MeanFarmWind(WakeFrame):

    def __init__(self, var_wd=FV.WD):
        super().__init__()
        self.var_wd = var_wd

    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):

        n_states = mdata.n_states
        stsel    = (np.arange(n_states), states_source_turbine)

        xyz   = fdata[FV.TXYH][stsel]
        delta = points - xyz[:, None, :] 
        del xyz

        wd = fdata[self.var_wd][stsel]

        nax  = np.zeros((n_states, 3, 3), dtype=FC.DTYPE)
        n    = nax[:, 0, :2]
        n[:] = wd2uv(wd, axis=-1)
        m    = nax[:, 1, :2]
        m[:] = np.stack([-n[:, 1], n[:, 0]], axis=-1)
        nax[:, 2, 2] = 1
        del wd

        coos = np.einsum('spd,sad->spa', delta, nax)

        return coos

        