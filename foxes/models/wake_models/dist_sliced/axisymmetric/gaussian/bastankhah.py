import numpy as np

from foxes.models.wake_models.dist_sliced.axisymmetric.gaussian.gaussian_wake_model import GaussianWakeModel
import foxes.variables as FV
import foxes.constants as FC

class BastankhahWake(GaussianWakeModel):

    def __init__(
            self, 
            superposition, 
            k=None,
            sbeta_factor=0.25, 
            ct_max=0.9999
        ):
        super().__init__(superpositions={FV.WS: superposition})

        self.k            = k
        self.ct_max       = ct_max
        self.sbeta_factor = sbeta_factor

    def init_wake_deltas(self, algo, fdata, n_points, wake_deltas):
        n_states = fdata.n_states
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def calc_amplitude_sigma_spsel(self, algo, fdata, states_source_turbine, x):

        # prepare:
        n_states = fdata.n_states
        n_points = x.shape[1]
        st_sel   = (np.arange(n_states), states_source_turbine)

        # get ct:
        ct    = np.zeros((n_states, n_points),dtype=FC.DTYPE)
        ct[:] = self.get_data(FV.CT, fdata)[st_sel][:, None]
        ct[ct>self.ct_max] = self.ct_max

        # select targets:
        sp_sel = (x > 0.) & (ct > 0.)
        if np.any(sp_sel):

            # apply selection:
            x  = x[sp_sel]
            ct = ct[sp_sel]

            # get D:
            D    = np.zeros((n_states, n_points),dtype=FC.DTYPE)
            D[:] = self.get_data(FV.D, fdata)[st_sel][:, None]
            D    = D[sp_sel]

            # get k:
            k    = np.zeros((n_states, n_points),dtype=FC.DTYPE)
            k[:] = self.get_data(FV.K, fdata, upcast="farm")[st_sel][:, None]
            k    = k[sp_sel]

            # calculate sigma:
            sbeta = np.sqrt( 0.5 * ( 1 + np.sqrt( 1. - ct ) ) / np.sqrt( 1. - ct ) )
            sblim = 1 / ( np.sqrt(8) * self.sbeta_factor )
            sbeta[sbeta > sblim] = sblim
            sigma = k * x + self.sbeta_factor * sbeta * D
            del x, k, sbeta, sblim

            # calculate amplitude:
            radicant     =  1. - ct / ( 8 * ( sigma / D )**2 ) 
            reals        = radicant>=0
            ampld        = -np.ones_like(radicant)
            ampld[reals] = np.sqrt(radicant[reals]) - 1.

        # case no targets:
        else:
            sp_sel = np.zeros_like(x, dtype=bool)
            n_sp   = np.sum(sp_sel)
            ampld  = np.zeros(n_sp, dtype=FC.DTYPE)
            sigma  = np.zeros(n_sp, dtype=FC.DTYPE)

        return {FV.WS: (ampld, sigma)}, sp_sel
