from abc import abstractmethod

from foxes.core.model import Model

class PartialWakesModel(Model):

    def __init__(self, wake_models=None, wake_frame=None):
        super().__init__()

        self.wake_models = wake_models
        self.wake_frame  = wake_frame

    def initialize(self, algo, farm_data):

        if self.wake_models is None:
            self.wake_models = algo.wake_models
        if self.wake_frame is None:
            self.wake_frame = algo.wake_frame

        if not self.wake_frame.initialized:
            self.wake_frame.initialize(algo, farm_data)
        for w in self.wake_models:
            if not w.initialized:
                w.initialize(algo, farm_data)

        super().initialize()

    @abstractmethod
    def n_wake_points(self,algo, fdata):
        pass
    
    def new_wake_deltas(self, algo, fdata):
        n_points    = self.n_wake_points(algo, fdata)
        wake_deltas = {}
        for w in self.wake_models:
            w.init_wake_deltas(algo, fdata, n_points, wake_deltas)
        return wake_deltas

    @abstractmethod
    def contribute_to_wake_deltas(self, algo, fdata, states_source_turbine, 
                                    wake_deltas):
        pass

    @abstractmethod
    def evaluate_results(self, algo, fdata, wake_deltas, states_turbine):
        pass
