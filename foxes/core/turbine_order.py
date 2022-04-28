
from foxes.core import FarmDataModel
import foxes.variables as FV

class TurbineOrder(FarmDataModel):

    def output_farm_vars(self, algo):
        return [FV.ORDER]
        