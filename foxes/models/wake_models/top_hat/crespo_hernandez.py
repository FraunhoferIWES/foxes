import numpy as np

from foxes.models.wake_models.top_hat.top_hat_wake_model import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC

class CrespoHernandezTIWake(TopHatWakeModel):

    def __init__(
            self, 
            superposition, 
            k=None, 
            use_ambti=False,
            sbeta_factor=0.25,
            near_wake_D=None,
            ct_max = 0.9999,
            a_near = 0.362,
            a_far  = 0.73,
            e1     = 0.83,
            e2     = -0.0325,
            e3     = -0.32,
        ):
        super().__init__(superpositions={FV.TI: superposition})

        self.k            = k
        self.ct_max       = ct_max
        self.a_near       = a_near
        self.a_far        = a_far
        self.e1           = e1
        self.e2           = e2
        self.e3           = e3
        self.use_ambti    = use_ambti
        self.sbeta_factor = sbeta_factor
        self.near_wake_D  = near_wake_D

    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        n_states = mdata.n_states
        wake_deltas[FV.TI] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def calc_wake_radius(self, algo, mdata, fdata, states_source_turbine, x, r, ct):
        
        # prepare:
        n_states = fdata.n_states
        st_sel   = (np.arange(n_states), states_source_turbine)

        # get D:
        D = fdata[FV.D][st_sel][:, None]

        # get k:
        k = self.get_data(FV.K, fdata, st_sel)
        if isinstance(k, np.ndarray):
            k = k[:, None]

        # calculate:
        sbeta  = np.sqrt(0.5 * ( 1 + np.sqrt( 1 - ct ) ) / np.sqrt( 1 - ct ))
        sblim  = 1 / ( np.sqrt(8) * self.sbeta_factor )
        sbeta[sbeta > sblim] = sblim
        radius = 4 * ( k * x + self.sbeta_factor * sbeta * D )
        
        return radius

    def calc_centreline_wake_deltas(self, algo, mdata, fdata, states_source_turbine,
                                        n_points, sp_sel, x, r, wake_r, ct):  

        # prepare:
        n_states = fdata.n_states
        n_targts = np.sum(sp_sel)
        st_sel   = (np.arange(n_states), states_source_turbine)
        TI       = FV.AMB_TI if self.use_ambti else FV.TI

        # read D from extra data:
        D    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][st_sel][:, None]
        D    = D[sp_sel]

        # get ti:
        ti    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ti[:] = fdata[TI][st_sel][:, None]
        ti    = ti[sp_sel]

        # prepare output:
        wake_deltas = np.zeros(n_targts, dtype=FC.DTYPE)

        # calc near wake length, if not given
        if self.near_wake_D is None:
            near_wake_D = ( 2**self.e1 * self.a_near / ( self.a_far * ti**self.e2 ) \
                            * ( 1 - np.sqrt( 1 - ct ) )**( 1 - self.e1 ) )**( 1 / self.e3 )
        else:
            near_wake_D = self.near_wake_D

        # calc near wake:
        sel = ( x < near_wake_D * D )
        if np.any(sel):
            wake_deltas[sel] = self.a_near * ( 1. - np.sqrt( 1. - ct[sel] ) )
        
        # calc far wake:
        if np.any(~sel):

            # calculate delta:
            #
            # Note the sign flip of the exponent ti[~sel]**(-0.0325)
            # compared to the original paper. This was found in 
            # https://doi.org/10.1016/j.jweia.2018.04.010, Eq. (46)
            # Without this flip the near and far wake areas are not
            # smoothly connected.
            #
            wake_deltas[~sel] =  self.a_far * ( ( 1. - np.sqrt( 1. - ct[~sel] ) ) / 2 )**self.e1 \
                                        * ti[~sel]**self.e2 * ( x[~sel] / D[~sel] )**self.e3

        return {FV.TI: wake_deltas}
