import numpy as np

from foxes.core import PartialWakesModel
import foxes.variables as FV

class RotorPoints(PartialWakesModel):

    def __init__(self, wake_models=None, wake_frame=None):
        super().__init__(wake_models, wake_frame)
    
    def initialize(self, algo):

        if not algo.rotor_model.initialized:
            algo.rotor_model.initialize(algo)
        
        self.WPOINTS = self.var("WPOINTS")

        super().initialize(algo)

    def get_wake_points(self, algo, mdata, fdata):
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape     
        return rpoints.reshape(n_states, n_turbines*n_rpoints, 3)

    def new_wake_deltas(self, algo, mdata, fdata):

        mdata[self.WPOINTS] = self.get_wake_points(algo, mdata, fdata)
        n_points = mdata[self.WPOINTS].shape[1]

        wake_deltas = {}
        for w in self.wake_models:
            w.init_wake_deltas(algo, mdata, fdata, n_points, wake_deltas)

        return wake_deltas

    def contribute_to_wake_deltas(self, algo, mdata, fdata, states_source_turbine, 
                                    wake_deltas):

        points = mdata[self.WPOINTS]
        wcoos  = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)

        for w in self.wake_models:
            w.contribute_to_wake_deltas(algo, mdata, fdata, states_source_turbine, 
                                            wcoos, wake_deltas)

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
            wdel[v] = d.reshape(n_states, n_turbines, n_rpoints)[st_sel]
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, wres, wdel)

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if update_amb_res:
                    mdata[FV.AMB_RPOINT_RESULTS][v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]
        
        algo.rotor_model.eval_rpoint_results(algo, mdata, fdata, wres, weights, 
                                                states_turbine=states_turbine)
