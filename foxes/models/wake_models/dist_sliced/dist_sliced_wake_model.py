from abc import abstractmethod

from foxes.core import WakeModel

class DistSlicedWakeModel(WakeModel):

    def __init__(self, superpositions):
        super().__init__()
        self.superp = superpositions
    
    def initialize(self, algo, data):
        super().initialize(algo, data)

        self.superp = {v: algo.mbook.wake_superpositions[s] for v, s in self.superp.items()} 
        for v, s in self.superp.items():
            if not s.initialized:
                s.initialize(algo, data)

    @abstractmethod
    def calc_xdata_spsel(self, algo, fdata, states_source_turbine, x):
        pass

    @abstractmethod
    def calc_wakes_ortho(self, algo, fdata, states_source_turbine, 
                            n_points, sp_sel, xdata, yz):
        pass

    def contribute_to_wake_deltas(self, algo, fdata, states_source_turbine, 
                                    wake_coos, wake_deltas):

        n_points = wake_coos.shape[1]

        x = wake_coos[:, :, 0]
        xdata, sp_sel = self.calc_xdata_spsel(algo, fdata, states_source_turbine, x)

        yz = wake_coos[:, :, 1:3][sp_sel]
        wdeltas = self.calc_wakes_ortho(algo, fdata, states_source_turbine,
                                            n_points, sp_sel, xdata, yz)
                
        for v, hdel in wdeltas.items():

            try:
                superp = self.superp[v]
            except KeyError:
                raise KeyError(f"Model '{self.name}': Missing wake superposition entry for variable '{v}', found {sorted(list(self.superp.keys()))}")

            wake_deltas[v] = superp.calc_wakes_plus_wake(
                                        algo, fdata, states_source_turbine,
                                        sp_sel, v, wake_deltas[v], hdel)

    def finalize_wake_deltas(self, algo, fdata, wake_deltas):
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(algo, fdata, v, wake_deltas[v])
        