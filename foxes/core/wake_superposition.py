from abc import abstractmethod

from foxes.core.model import Model

class WakeSuperposition(Model):
    
    @abstractmethod
    def calc_wakes_plus_wake(self, algo, mdata, fdata, states_source_turbine,
                                sel_sp, variable, wake_delta, wake_model_result):
        pass

    @abstractmethod
    def calc_final_wake_delta(self, algo, mdata, fdata, variable, wake_delta):
        pass
