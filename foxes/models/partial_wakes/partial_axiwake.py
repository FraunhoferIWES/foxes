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

        # get normalized 2D vector between rotor and wake centres:
        sel = (R > 0.)
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            temp = n[~sel]
            temp[:, 0] = 1.
            n[~sel] = temp
            del temp

        # case wake centre outside rotor disk:
        sel = (R >= D/2)
        if np.any(sel):
            steps  = np.linspace(0., 1., self.n_steps, endpoint=False)
            steps  = steps + steps[1] / 2
            steps -= 0.5
            r[sel] = R[sel] + D[sel] * steps[None, :]
        
        # case wake centre inside rotor disk:
        if np.any(~sel):
            d       = R[~sel] + D[~sel]
            steps   = np.linspace(0., 1., self.n_steps, endpoint=False)
            r[~sel] = d[:, None] * steps[None, :]
        
        print("PAWAKE",x[0], r[0])
        for w in self.wake_models:
            wdeltas, sp_sel = w.calc_wakes_spsel_x_r(algo, mdata, fdata, 
                                                        states_source_turbine, x, r)
            print(wdeltas, wake_deltas)

        quit()

        for w in self.wake_models:

            xdata, sp_sel = w.calc_xdata_spsel(algo, mdata, fdata, states_source_turbine, x)

            r = np.linalg.norm(wcoos[:, :, :, 1:3], axis=-1)
            
            TODO
        

        



        n   = wcoos[:, :, :, 1:3]
        R   = np.linalg.norm(n, axis=-1)
        sel = (R > D[:, :, None])
        if np.any(sel):
            n[sel] /= R[sel][:, None]
        if np.any(~sel):
            temp = n[~sel]
            temp[:, 0] = 1.
            n[~sel] = temp
            del temp
        steps  = np.linspace(0., 1., self.n_steps, endpoint=False)
        steps  = steps + steps[1] / 2
        steps -= 0.5
        wcoos[:, :, :, 1:3] += D[:, :, None, None] * steps[None, None, :, None] * n
        print("WCOOS B",steps, D[0,0], n[0,0], wcoos[0,0])
        wcoos = wcoos.reshape(n_states, n_points, 3)
        mdata[self.R] = R
        del n, sel, steps, R

        for w in self.wake_models:
            w.contribute_to_wake_deltas(algo, mdata, fdata, states_source_turbine, 
                                            wcoos, wake_deltas)

    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine):

        n_states   = mdata.n_states
        n_turbines = algo.n_turbines
        n_wpoints  = self.n_wake_points(algo, mdata, fdata)
        D          = fdata[FV.D]
        R          = mdata[self.R]

        A     = np.zeros((n_states, n_turbines, self.n_steps + 1), dtype=FC.DTYPE)
        steps = np.linspace(0., 1., self.n_steps + 1, endpoint=True) - 0.5
        R1    = np.zeros((n_states, n_turbines, self.n_steps), dtype=FC.DTYPE)
        R1[:] = D[:, :, None]/2
        R2    = R + D[:, :, None] * steps[None, None, 1:]
        A[:, :, 1:] = calc_area(R1, R2, R)
        print("AWKE")
        print(steps)
        print("R =", R[0,0])
        print(A[0,0])
        A           = A[:, :, 1:] - A[:, :, :-1]
        print(A[0,0])
        A          /= np.sum(A, axis=-1)[:, :, None]
        sweights    = A
        del A, steps, R, R1, R2, D

        wdel   = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, d in wake_deltas.items():
            wdel[v] = np.einsum('stw,stw->st', d.reshape(n_states, n_turbines, n_wpoints), sweights)
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wdel)

        weights = self.get_data(FV.RWEIGHTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        __, __, n_rpoints, __ = rpoints.shape

        wres = {}
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]
            if v in wake_deltas:
                wres[v] += wdel[v]
            wres[v] = wres[v][:, None]
        
        self.rotor_model.eval_rpoint_results(algo, mdata, fdata, wres, weights, 
                                                states_turbine=states_turbine)
