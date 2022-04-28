from abc import ABCMeta, abstractmethod

class WakeSuperposition(metaclass=ABCMeta):

    def __init__(self):
        self.name = type(self).__name__ # updated by algorithm   
        self.__initialized = False
    
    def initialize(self, algo, farm_data):
        self.__initialized = True
    
    @property
    def initialized(self):
        return self.__initialized
    
    @abstractmethod
    def calc_wakes_plus_wake(self, algo, fdata, states_source_turbine,
                                sel_sp, variable, wake_delta, wake_model_result):
        pass

    @abstractmethod
    def calc_final_wake_delta(self, algo, fdata, variable, wake_delta):
        pass

    def finalize(self, algo, farm_data):
        pass
