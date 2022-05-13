import numpy as np

from foxes.core import PartialWakesModel
from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel
from foxes.models.rotor_models.grid import GridRotor
import foxes.variables as FV

class PartialDistSlicedWake(PartialWakesModel):

    def __init__(self, n=None, wake_models=None, wake_frame=None, rotor_model=None, **kwargs):
        super().__init__(wake_models, wake_frame)

        self.rotor_model = rotor_model
        self.grotor      = None if n is None else GridRotor(n=n, calc_vars=[], **kwargs)

    def initialize(self, algo):
        super().initialize(algo)

        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model
        if not self.rotor_model.initialized:
            self.rotor_model.initialize(algo)
            
        for w in self.wake_models:
            if not isinstance(w, DistSlicedWakeModel):
                raise TypeError(f"Partial wakes '{self.name}': Cannot be applied to wake model '{w.name}', since not an DistSlicedWakeModel")

        if self.grotor is None:
            self.grotor = self.rotor_model
        else:
            self.grotor.name = f"{self.name}_grotor"
            self.grotor.initialize(algo)

        self.YZ = self.var("YZ")
        self.W  = self.var(FV.WEIGHT)

    def n_wake_points(self, algo, mdata, fdata):
        return algo.n_turbines

    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                                    states_source_turbine, wake_deltas):

        # calc coordinates to rotor centres:
        wcoos = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, 
                                                fdata[FV.TXYH])

        # get x coordinates:
        x = wcoos[:, :, 0]
        del wcoos

        # evaluate grid rotor:
        n_states   = fdata.n_states
        n_turbines = fdata.n_turbines
        n_rpoints  = self.grotor.n_rotor_points()
        points     = self.grotor.get_rotor_points(algo, mdata, fdata).reshape(n_states, n_turbines*n_rpoints, 3)
        wcoos      = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)
        yz         = wcoos.reshape(n_states, n_turbines, n_rpoints, 3)[:, :, :, 1:3]
        weights    = self.grotor.rotor_point_weights()
        del points, wcoos

        # evaluate wake models:
        for w in self.wake_models:

            wdeltas, sp_sel = w.calc_wakes_spsel_x_yz(algo, mdata, fdata, 
                                                        states_source_turbine, x, yz)
            
            for v, wdel in wdeltas.items():

                d = np.einsum('ps,s->p', wdel, weights)

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

    def finalize(self, algo, clear_mem=False):
        self.grotor.finalize(algo, clear_mem=clear_mem)
        super().finalize(algo, clear_mem=clear_mem)
