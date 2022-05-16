import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced.axisymmetric.axisymmetric_wake_model import AxisymmetricWakeModel

class GaussianWakeModel(AxisymmetricWakeModel):

    @abstractmethod
    def calc_amplitude_sigma_spsel(self, algo, mdata, fdata, states_source_turbine, x):
        pass

    def calc_wakes_spsel_x_r(self, algo, mdata, fdata, states_source_turbine, x, r):

        amsi, sp_sel = self.calc_amplitude_sigma_spsel(algo, mdata, fdata, 
                                                         states_source_turbine, x)
        wdeltas = {}
        rsel    = r[sp_sel]
        for v in amsi.keys():
            ampld, sigma = amsi[v]
            wdeltas[v]   = ampld[:, None] * np.exp(-0.5 * (rsel/sigma[:, None])**2)

        return wdeltas, sp_sel
