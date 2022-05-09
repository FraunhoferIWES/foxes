import numpy as np

from foxes.core import PartialWakesModel
from foxes.models.wake_models.dist_sliced.axisymmetric import AxisymmetricWakeModel
from foxes.tools.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

class PartialAxiwake(PartialWakesModel):

    def __init__(self, n_steps, wake_models=None, wake_frame=None, rotor_model=None):
        super().__init__(wake_models, wake_frame)

        self.n_steps     = n_steps
        self.rotor_model = rotor_model

    def initialize(self, algo):
        super().initialize(algo)

        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model
        if not self.rotor_model.initialized:
            self.rotor_model.initialize(algo)
            
        for w in self.wake_models:
            if not isinstance(w, AxisymmetricWakeModel):
                raise TypeError(f"Partial wakes '{self.name}': Cannot be applied to wake model '{w.name}', since not an AxisymmetricWakeModel")

        self.R = self.var("R")

    def n_wake_points(self, algo, mdata, fdata):
        return algo.n_turbines

    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                                    states_source_turbine, wake_deltas):

        # prepare:
        n_states   = mdata.n_states
        n_turbines = algo.n_turbines
        D          = fdata[FV.D]

        # calc coordinates to rotor centres:
        wcoos = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, 
                                                fdata[FV.TXYH])

        # prepare x and r coordinates:
        x  = wcoos[:, :, 0]
        n  = wcoos[:, :, 1:3]
        R  = np.linalg.norm(n, axis=-1)
        r  = np.zeros((n_states, n_turbines, self.n_steps), dtype=FC.DTYPE)
        del wcoos

        # prepare circle section area calculation:
        A       = np.zeros((n_states, n_turbines, self.n_steps), dtype=FC.DTYPE)
        weights = np.zeros_like(A)

        # get normalized 2D vector between rotor and wake centres:
        sel = (R > 0.)
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            n[:, :, 0][~sel] = 1

        # case wake centre outside rotor disk:
        sel = (x > 1e-5) & (R > D/2)
        if np.any(sel):

            n_sel   = np.sum(sel)
            Rsel    = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            Rsel[:] = R[sel][:, None]
            Dsel    = D[sel][:, None]

            # equal delta R2:
            R1     = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            R1[:]  = Dsel / 2
            steps  = np.linspace(0., 1., self.n_steps + 1, endpoint=True) - 0.5
            R2     = np.zeros_like(R1)
            R2[:]  = Rsel + Dsel * steps[None, :]
            r[sel] = 0.5 * ( R2[:, 1:] + R2[:, :-1] )
    
            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1]
            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, Rsel, Dsel, R1, R2
        
        # case wake centre inside rotor disk:
        sel = (x > 1e-5) & (R < D/2)
        if np.any(sel):

            n_sel   = np.sum(sel)
            Rsel    = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            Rsel[:] = R[sel][:, None]
            Dsel    = D[sel][:, None]

            # equal delta R2:
            R1        = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            R1[:, 1:] = Dsel / 2
            R2        = np.zeros_like(R1)
            R2[:, 1:] = Rsel[:, :-1] + Dsel/2
            R2[:]    *= np.linspace(0., 1, self.n_steps + 1, endpoint=True)[None, :]
            hr        = 0.5 * ( R2[:, 1:] + R2[:, :-1] )
            hr[:, 0]  = 0.
            r[sel]    = hr
        
            """
            # equal delta r:
            # seems to perform worse than equal delta R2
            steps     = np.linspace(0., 1., self.n_steps, endpoint=False)
            r[sel]    = ( Rsel[:, :-1] + Dsel/2 ) * steps[None, :]
            hr        = r[sel]
            R1        = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            R1[:, 1:] = Dsel / 2
            R2        = np.zeros_like(R1)
            R2[:, 1:-1] = 0.5 * ( hr[:, 1:] + hr[:, :-1] )
            R2[:, -1]   = Rsel[:, -1] + Dsel[:,-1]/2
            """

            """ 
            # equal weights:
            # seems to perform worse than equal delta R2
            R1        = np.zeros((n_sel, self.n_steps + 1), dtype=FC.DTYPE)
            R1[:, 1:] = Dsel / 2
            R2        = np.zeros_like(R1)
            R2[:, 1:] = ( Rsel[:, :-1] + Dsel/2 ) / np.sqrt(self.n_steps)
            R2[:]    *= np.sqrt(np.linspace(0., self.n_steps, self.n_steps + 1, endpoint=True))[None, :]
            hr        = 0.5 * ( R2[:, 1:] + R2[:, :-1] )
            hr[:, 0]  = 0.
            r[sel]   = hr
            """

            hA = calc_area(R1, R2, Rsel)
            hA = hA[:, 1:] - hA[:, :-1]
            weights[sel] = hA / np.sum(hA, axis=-1)[:, None]
            del hA, hr, Rsel, Dsel, R1, R2

        # evaluate wake models:
        for w in self.wake_models:

            wdeltas, sp_sel = w.calc_wakes_spsel_x_r(algo, mdata, fdata, 
                                                        states_source_turbine, x, r)
            
            for v, wdel in wdeltas.items():

                d = np.einsum('ps,ps->p', wdel, weights[sp_sel])
                
                try:
                    superp = w.superp[v]
                except KeyError:
                    raise KeyError(f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{w.name}', found {sorted(list(w.superp.keys()))}")

                wake_deltas[v] = superp.calc_wakes_plus_wake(algo, mdata, fdata, states_source_turbine, 
                                                            sp_sel, v, wake_deltas[v], d)


    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine):

        weights = self.get_data(FV.RWEIGHTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        wdel   = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wdel)

        wres = {}
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]
            if v in wake_deltas:
                wres[v] += wdel[v]
            wres[v] = wres[v][:, None]
        
        self.rotor_model.eval_rpoint_results(algo, mdata, fdata, wres, weights, 
                                                states_turbine=states_turbine)