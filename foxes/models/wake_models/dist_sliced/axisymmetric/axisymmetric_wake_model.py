import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced.dist_sliced_wake_model import DistSlicedWakeModel

class AxisymmetricWakeModel(DistSlicedWakeModel):

    @abstractmethod
    def calc_wakes_spsel_x_r(self, algo, mdata, fdata, tates_source_turbine, x, r):
        pass

    def calc_wakes_spsel_x_yz(self, algo, mdata, fdata, tates_source_turbine, x, yz):
        r = np.linalg.norm(yz, axis=-1)
        return self.calc_wakes_spsel_x_r(algo, mdata, fdata, tates_source_turbine, x, r)
