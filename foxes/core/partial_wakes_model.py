from abc import abstractmethod

from foxes.core.model import Model
from foxes.tools import all_subclasses

class PartialWakesModel(Model):

    def __init__(self, wake_models=None, wake_frame=None):
        super().__init__()

        self.wake_models = wake_models
        self.wake_frame  = wake_frame

    def initialize(self, algo):

        if self.wake_models is None:
            self.wake_models = algo.wake_models
        if self.wake_frame is None:
            self.wake_frame = algo.wake_frame

        if not self.wake_frame.initialized:
            self.wake_frame.initialize(algo)
        for w in self.wake_models:
            if not w.initialized:
                w.initialize(algo)

        super().initialize(algo)
    
    @abstractmethod
    def new_wake_deltas(self, algo, mdata, fdata):
        pass

    @abstractmethod
    def contribute_to_wake_deltas(self, algo, mdata, fdata, 
                            states_source_turbine, wake_deltas):
        pass

    @abstractmethod
    def evaluate_results(self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=False):
        pass

    @classmethod
    def new(cls, pwake_type, **kwargs):
        """
        Run-time partial wakes factory.

        Parameters
        ----------
        pwake_type : str
            The selected derived class name

        """

        if pwake_type is None:
            return None

        allc  = all_subclasses(cls)
        found = pwake_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == pwake_type:
                    return scls(**kwargs)

        else:
            estr = "Partial wakes model type '{}' is not defined, available types are \n {}".format(
                pwake_type, sorted([ i.__name__ for i in allc]))
            raise KeyError(estr)