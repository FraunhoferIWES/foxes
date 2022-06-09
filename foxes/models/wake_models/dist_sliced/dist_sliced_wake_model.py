from abc import abstractmethod

from foxes.core import WakeModel

class DistSlicedWakeModel(WakeModel):

    def __init__(self, superpositions):
        super().__init__()
        self.superp = superpositions
    
    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        self.superp = {v: algo.mbook.wake_superpositions[s] for v, s in self.superp.items()} 
        for v, s in self.superp.items():
            if not s.initialized:
                s.initialize(algo, verbosity=verbosity)

    @abstractmethod
    def calc_wakes_spsel_x_yz(self, algo, mdata, fdata, states_source_turbine, x, yz):
        pass

    def contribute_to_wake_deltas(self, algo, mdata, fdata, states_source_turbine, 
                                    wake_coos, wake_deltas):
        
        x  = wake_coos[:, :, 0]
        yz = wake_coos[:, :, None, 1:3]

        wdeltas, sp_sel = self.calc_wakes_spsel_x_yz(algo, mdata, fdata, 
                                                        states_source_turbine, x, yz)
        for v, hdel in wdeltas.items():

            try:
                superp = self.superp[v]
            except KeyError:
                raise KeyError(f"Model '{self.name}': Missing wake superposition entry for variable '{v}', found {sorted(list(self.superp.keys()))}")

            wake_deltas[v] = superp.calc_wakes_plus_wake(
                                        algo, mdata, fdata, states_source_turbine,
                                        sp_sel, v, wake_deltas[v], hdel[:, 0])

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(algo, mdata, fdata, v,
                                                        amb_results[v], wake_deltas[v])
        