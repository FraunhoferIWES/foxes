
from abc import abstractmethod

from foxes.core.farm_data_model import FarmDataModel

class TurbineModel(FarmDataModel):

    def initialize(self, algo, farm_data, st_sel):
        super().initialize(algo, farm_data)

    @abstractmethod
    def calculate(self, algo, fdata, st_sel):
        pass

    def finalize(self, algo, farm_data, st_sel):
        pass
