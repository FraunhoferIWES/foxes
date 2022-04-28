from abc import abstractmethod

from foxes.core.model import Model

class WakeModel(Model):
    
    def initialize(self, algo, farm_data):
        super().initialize()
    
    @abstractmethod
    def init_wake_deltas(self, algo, fdata, n_points, wake_deltas):
        pass

    @abstractmethod
    def contribute_to_wake_deltas(self, algo, fdata, states_source_turbine, 
                                    wake_coos, wake_deltas):
        pass

    def finalize_wake_deltas(self, algo, fdata, wake_deltas):
        pass

    def finalize(self, algo, farm_data):
        pass

