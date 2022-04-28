import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced.dist_sliced_wake_model import DistSlicedWakeModel

class AxisymmetricWakeModel(DistSlicedWakeModel):

    @abstractmethod
    def calc_wakes_radial(self, algo, fdata, states_source_turbine, 
                            n_points, sp_sel, xdata, r):
        pass

    def calc_wakes_ortho(self, algo, fdata, states_source_turbine, 
                            n_points, sp_sel, xdata, yz):
        
        r = np.linalg.norm(yz, axis=-1)

        return self.calc_wakes_radial(algo, fdata, states_source_turbine, 
                                            n_points, sp_sel, xdata, r)
