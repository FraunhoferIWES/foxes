import numpy as np

from foxes.core import PartialWakesModel
import foxes.variables as FV

class RotorPoints(PartialWakesModel):

    def __init__(self, wake_models=None, wake_frame=None, rotor_model=None):
        super().__init__(wake_models, wake_frame)
        self.rotor_model = rotor_model
    
    def initialize(self, algo):

        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model
        if not self.rotor_model.initialized:
            self.rotor_model.initialize(algo)
        
        self.WPOINTS = self.var("WPOINTS")

        super().initialize(algo)

    def n_wake_points(self, algo, mdata, fdata):
        __, n_turbines, n_rpoints, __ = self.get_data(FV.RPOINTS, mdata).shape
        return n_turbines * n_rpoints

    def get_wake_points(self, algo, mdata, fdata):
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape
        return rpoints.reshape(n_states, n_turbines*n_rpoints, 3)

    def new_wake_deltas(self, algo, mdata, fdata):
        mdata[self.WPOINTS] = None
        return super().new_wake_deltas(algo, mdata, fdata)

    def contribute_to_wake_deltas(self, algo, mdata, fdata, states_source_turbine, 
                                    wake_deltas):
        
        if mdata[self.WPOINTS] is None:
            mdata[self.WPOINTS] = self.get_wake_points(algo, mdata, fdata)
        
        points = mdata[self.WPOINTS]
        wcoos  = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)

        for w in self.wake_models:
            w.contribute_to_wake_deltas(algo, mdata, fdata, states_source_turbine, 
                                            wcoos, wake_deltas)

    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine):
        
        weights = self.get_data(FV.RWEIGHTS, mdata)
        amb_res = self.get_data(FV.AMB_RPOINT_RESULTS, mdata)
        rpoints = self.get_data(FV.RPOINTS, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        wdel   = {}
        st_sel = (np.arange(n_states), states_turbine)
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, n_rpoints)[st_sel]
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