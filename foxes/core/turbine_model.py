
from abc import abstractmethod

from foxes.core.farm_data_model import FarmDataModel

class TurbineModel(FarmDataModel):

    def initialize(self, algo, st_sel):
        super().initialize(algo)

    @abstractmethod
    def calculate(self, algo, mdata, fdata, st_sel):
        pass

    def finalize(self, algo, st_sel, clear_mem=False):
        super().finalize(algo, clear_mem)
