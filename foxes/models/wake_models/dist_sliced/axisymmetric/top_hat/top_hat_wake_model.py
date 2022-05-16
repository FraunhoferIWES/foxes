import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced.axisymmetric.axisymmetric_wake_model import AxisymmetricWakeModel
import foxes.variables as FV
import foxes.constants as FC

class TopHatWakeModel(AxisymmetricWakeModel):

    @abstractmethod
    def calc_wake_radius(self, algo, mdata, fdata, states_source_turbine, x, ct):
        pass

    @abstractmethod
    def calc_centreline_wake_deltas(self, algo, mdata, fdata, states_source_turbine,
                                        n_points, sp_sel, x, wake_r, ct):
        pass

    def calc_wakes_spsel_x_r(self, algo, mdata, fdata, states_source_turbine, x, r):

        n_states = mdata.n_states
        n_points = x.shape[1]
        st_sel   = (np.arange(n_states), states_source_turbine)

        ct    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = fdata[FV.CT][st_sel][:, None]

        wake_r = self.calc_wake_radius(algo, mdata, fdata, states_source_turbine, x, ct)

        wdeltas = {}
        sp_sel  = (ct > 0.) & (x > 1e-5) & np.any(r < wake_r[:, :, None], axis=2)
        if np.any(sp_sel):

            x      = x[sp_sel]
            r      = r[sp_sel]
            ct     = ct[sp_sel]
            wake_r = wake_r[sp_sel]

            cl_del = self.calc_centreline_wake_deltas(algo, mdata, fdata, states_source_turbine,
                                                        n_points, sp_sel, x, wake_r, ct)
            
            nsel = (r >= wake_r[:, None])
            for v, wdel in cl_del.items():
                wdeltas[v]       = np.zeros_like(r)
                wdeltas[v][:]    = wdel[:, None]
                wdeltas[v][nsel] = 0.

        return wdeltas, sp_sel

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(algo, mdata, fdata, v, 
                                                        amb_results[v], wake_deltas[v])
        