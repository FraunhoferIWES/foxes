
from foxes.models.partial_wakes.distsliced import PartialDistSlicedWake
from foxes.models.rotor_models.grid import GridRotor

class PartialGrid(PartialDistSlicedWake):

    def __init__(self, n, wake_models=None, wake_frame=None, rotor_model=None, **kwargs):
        super().__init__(n, wake_models, wake_frame, rotor_model, **kwargs)

        if not isinstance(self.grotor, GridRotor):
            raise ValueError(f"Wrong grotor type, expecting {GridRotor.__name__}, got {type(self.grotor).__name__}")

    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                                    states_source_turbine, wake_deltas):

        # evaluate grid rotor:
        n_states   = fdata.n_states
        n_turbines = fdata.n_turbines
        n_rpoints  = self.grotor.n_rotor_points()
        n_points   = n_turbines * n_rpoints
        points     = self.grotor.get_rotor_points(algo, mdata, fdata).reshape(n_states, n_points, 3)
        wcoos      = self.wake_frame.get_wake_coos(algo, mdata, fdata, states_source_turbine, points)
        del points

        # evaluate wake models:
        for w in self.wake_models:
            w.contribute_to_wake_deltas(algo, mdata, fdata, states_source_turbine, 
                                            wcoos, wake_deltas)                                                            
