from abc import abstractmethod

from foxes.core.point_data_model import PointDataModel
import foxes.variables as FV

class States(PointDataModel):

    @abstractmethod
    def size(self):
        pass

    def index(self):
        pass

    @abstractmethod
    def weights(self, algo):
        pass

    def model_input_data(self, algo):

        idata = {"coords": {}, "data_vars": {}}

        sinds = self.index()
        if sinds is not None:
            idata["coords"][FV.STATE] = sinds

        weights = self.weights(algo)
        if len(weights.shape) != 2:
            raise ValueError(f"States '{self.name}': Wrong weights dimension, expecing ({FV.STATE}, {FV.TURBINE}), got shape {weights.shape}")
        if weights.shape[1] != algo.n_turbines:
            raise ValueError(f"States '{self.name}': Wrong size of second axis dimension '{FV.TURBINE}': Expecting {self.n_turbines}, got {weights.shape[1]}")
        idata["data_vars"][FV.WEIGHT] = ((FV.STATE, FV.TURBINE), weights)

        return idata

    def output_point_vars(self, algo):
        return [FV.WS, FV.WD, FV.TI, FV.RHO]
