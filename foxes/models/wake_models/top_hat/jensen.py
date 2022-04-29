import numpy as np

from foxes.models.wake_models.top_hat.top_hat_wake_model import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC

class JensenWake(TopHatWakeModel):

    def __init__(self, superposition, k=None, ct_max=0.9999):
        super().__init__(superpositions={FV.WS: superposition})

        self.k      = k
        self.ct_max = ct_max

    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        n_states = mdata.n_states
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def calc_wake_radius(self, algo, mdata, fdata, states_source_turbine, x, r, ct):

        n_states = mdata.n_states
        st_sel   = (np.arange(n_states), states_source_turbine)

        R = fdata[FV.D][st_sel][:, None] / 2
        k = self.get_data(FV.K, fdata, st_sel)

        if isinstance(k, np.ndarray):
            k = k[:, None]

        return R + k * x

    def calc_centreline_wake_deltas(self, algo, mdata, fdata, states_source_turbine,
                                        n_points, sp_sel, x, r, wake_r, ct):

        n_states = mdata.n_states
        st_sel   = (np.arange(n_states), states_source_turbine)

        R    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        R[:] = fdata[FV.D][st_sel][:, None] / 2
        R    = R[sp_sel]

        return {
            FV.WS: -( R / wake_r )**2 * ( 1. - np.sqrt( 1. - ct ) )
        }
