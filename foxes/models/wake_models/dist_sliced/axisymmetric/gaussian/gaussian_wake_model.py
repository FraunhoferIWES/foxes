import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced.axisymmetric.axisymmetric_wake_model import AxisymmetricWakeModel

class GaussianWakeModel(AxisymmetricWakeModel):

    @abstractmethod
    def calc_amplitude_sigma_spsel(self, algo, mdata, fdata, states_source_turbine, x):
        pass

    def calc_xdata_spsel(self, algo, mdata, fdata, states_source_turbine, x):
        return self.calc_amplitude_sigma_spsel(algo, mdata, fdata, states_source_turbine, x)

    def calc_wakes_radial(self, algo, mdata, fdata, states_source_turbine, 
                            n_points, sp_sel, xdata, r):

        out = {}
        for v in xdata.keys():
            ampld, sigma = xdata[v]
            out[v] = ampld * np.exp(-0.5 * (r/sigma)**2)
        
        return out
