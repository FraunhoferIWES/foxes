import numpy as np
from abc import abstractmethod

from foxes.core import WakeModel
import foxes.variables as FV
import foxes.constants as FC

class TopHatWakeModel(WakeModel):

    def __init__(self, superpositions):
        super().__init__()
        self.superp = superpositions
    
    def initialize(self, algo):
        super().initialize(algo)

        self.superp = {v: algo.mbook.wake_superpositions[s] for v, s in self.superp.items()} 
        for v, s in self.superp.items():
            if not s.initialized:
                s.initialize(algo)

    @abstractmethod
    def calc_wake_radius(self, algo, mdata, fdata, states_source_turbine, x, r, ct):
        pass

    @abstractmethod
    def calc_centreline_wake_deltas(self, algo, mdata, fdata, states_source_turbine,
                                        n_points, sp_sel, x, r, wake_r, ct):
        pass

    def contribute_to_wake_deltas(self, algo, mdata, fdata, states_source_turbine, 
                                    wake_coos, wake_deltas):

        n_states = mdata.n_states
        n_points = wake_coos.shape[1]
        st_sel   = (np.arange(n_states), states_source_turbine)

        ct    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        ct[:] = fdata[FV.CT][st_sel][:, None]

        x  = wake_coos[:, :, 0]
        r  = np.linalg.norm(wake_coos[:, :, 1:3], axis=-1)

        wake_r = self.calc_wake_radius(algo, mdata, fdata, states_source_turbine, x, r, ct)

        sp_sel = (ct > 0.) & (x > 0.) & (r < wake_r)
        if np.any(sp_sel):

            x      = x[sp_sel]
            r      = r[sp_sel]
            ct     = ct[sp_sel]
            wake_r = wake_r[sp_sel]

            wake_d = self.calc_centreline_wake_deltas(algo, mdata, fdata, states_source_turbine,
                                                        n_points, sp_sel, x, r, wake_r, ct)
            del x, r, ct, wake_r
            
            for v, d in wake_d.items():
                try:
                    superp = self.superp[v]
                except KeyError:
                    raise KeyError(f"Model '{self.name}': Missing wake superposition entry for variable '{v}', found {sorted(list(self.superp.keys()))}")

                wake_deltas[v] = superp.calc_wakes_plus_wake(
                                            algo, mdata, fdata, states_source_turbine,
                                            sp_sel, v, wake_deltas[v], d)

    def finalize_wake_deltas(self, algo, mdata, fdata, wake_deltas):
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(algo, mdata, fdata, v, wake_deltas[v])
        