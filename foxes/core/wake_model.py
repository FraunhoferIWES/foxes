from abc import abstractmethod

from foxes.core.model import Model

class WakeModel(Model):
    
    @abstractmethod
    def init_wake_deltas(self, algo, mdata, fdata, n_points, wake_deltas):
        pass

    @abstractmethod
    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                            states_source_turbine, wake_coos, wake_deltas):
        pass

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        pass
