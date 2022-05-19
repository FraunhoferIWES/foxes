import numpy as np

from foxes.core import PartialWakesModel
from foxes.models.wake_models.dist_sliced.axisymmetric.top_hat.top_hat_wake_model import TopHatWakeModel
from foxes.tools.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

class PartialTopHat(PartialWakesModel):

    def __init__(self, wake_models=None, wake_frame=None, rotor_model=None):
        super().__init__(wake_models, wake_frame)
        self.rotor_model = rotor_model

    def initialize(self, algo):
        super().initialize(algo)

        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model
        if not self.rotor_model.initialized:
            self.rotor_model.initialize(algo)
            
        for w in self.wake_models:
            if not isinstance(w, TopHatWakeModel):
                raise TypeError(f"Partial wakes '{self.name}': Cannot be applied to wake model '{w.name}', since not a TopHatWakeModel")

    def get_wake_points(self, algo, mdata, fdata):
        return fdata[FV.TXYH]

    def new_wake_deltas(self, algo, mdata, fdata):

        n_points    = fdata.n_turbines
        wake_deltas = {}
        for w in self.wake_models:
            w.init_wake_deltas(algo, mdata, fdata, n_points, wake_deltas)

        return wake_deltas

    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                                    states_source_turbine, wake_deltas):

        n_states = mdata.n_states
        n_points = fdata.n_turbines
        stsel    = (np.arange(n_states), states_source_turbine)

        if FV.WCOOS_ID not in mdata or mdata[FV.WCOOS_ID] != states_source_turbine[0]:
            points = self.get_wake_points(algo, mdata, fdata)
            wcoos  = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)
            mdata[FV.WCOOS_ID] = states_source_turbine[0]
            mdata[FV.WCOOS_X]  = wcoos[:, :, 0]
            mdata[FV.WCOOS_R]  = np.linalg.norm(wcoos[:, :, 1:3], axis=-1)
            wcoos[:, :, 1:3]   = 0
            del points
        else:
            wcoos = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            wcoos[:, :, 0] = mdata[FV.WCOOS_X]
        
        ct    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = fdata[FV.CT][stsel][:, None]
        x     = mdata[FV.WCOOS_X]

        sel0 = (ct > 0.) & (x > 0.)
        if np.any(sel0):

            R  = mdata[FV.WCOOS_R]
            r  = np.zeros_like(R)
            D  = fdata[FV.D]

            for w in self.wake_models:

                wr = w.calc_wake_radius(algo, mdata, fdata, states_source_turbine, x, ct)

                sel_sp = sel0 & (wr > R - D/2) 
                if np.any(sel_sp):

                    hx  = x[sel_sp]
                    hct = ct[sel_sp]
                    hwr = wr[sel_sp]

                    clw = w.calc_centreline_wake_deltas(algo, mdata, fdata, states_source_turbine,
                                                            n_points, sel_sp, hx, hwr, hct)
                    del hx, hct

                    hR = R[sel_sp]
                    hD = D[sel_sp]
                    weights = calc_area(hD/2, hwr, hR) / ( np.pi * (hD/2)**2 )
                    del hD, hwr, hR

                    for v, d in clw.items():
                        try:
                            superp = w.superp[v]
                        except KeyError:
                            raise KeyError(f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{w.name}', found {sorted(list(w.superp.keys()))}")

                        wake_deltas[v] = superp.calc_wakes_plus_wake(algo, mdata, fdata, states_source_turbine, 
                                                                    sel_sp, v, wake_deltas[v], weights*d)

    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=False):
        
        weights = self.get_data(FV.RWEIGHTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        wres   = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, ares in amb_res.items():
            wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        del amb_res

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, 1)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if update_amb_res:
                    mdata[FV.AMB_RPOINT_RESULTS][v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]
        
        self.rotor_model.eval_rpoint_results(algo, mdata, fdata, wres, weights, 
                                                states_turbine=states_turbine)
                                                