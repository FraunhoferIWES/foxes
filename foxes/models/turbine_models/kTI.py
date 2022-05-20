import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC

class kTI(TurbineModel):

    def __init__(self, kTI=None, ti_var=FV.TI, ti_val=None):
        super().__init__()

        self.ti_var = ti_var
        self.__dict__[FV.KTI] = kTI
        self.__dict__[ti_var] = ti_val

    def output_farm_vars(self, algo):
        return [FV.K]
    
    def calculate(self, algo, mdata, fdata, st_sel):

        kTI = self.get_data(FV.KTI, fdata, st_sel)
        ti  = self.get_data(self.ti_var, fdata, st_sel)

        k = fdata.get(FV.K, 
                np.zeros((fdata.n_states, fdata.n_turbines), dtype=FC.DTYPE))
                
        k[st_sel] = kTI * ti

        return {FV.K: k}  
