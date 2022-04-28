from abc import abstractmethod

from foxes.core.model import Model

class WakeFrame(Model):
        
    def initialize(self, algo, farm_data):
        super().initialize()

    @abstractmethod
    def get_wake_coos(self, algo, fdata, states_source_turbine, points):
        pass

    def finalize(self, algo, farm_data):
        pass
