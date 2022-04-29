from abc import abstractmethod

from foxes.core.model import Model

class WakeFrame(Model):

    @abstractmethod
    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):
        pass

