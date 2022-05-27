
from foxes.core import FarmDataModel
import foxes.variables as FV

class TurbineOrder(FarmDataModel):
    """
    Abstract base class for turbine order models.

    Turbine orders define the order of turbine wake
    calculations for each state.
    
    """

    def output_farm_vars(self, algo):
        return [FV.ORDER]
        