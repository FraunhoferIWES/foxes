import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC

class kTI(TurbineModel):

    def __init__(self, kTI=None, ti=None):
        super().__init__()

        self.__dict__[FV.KTI] = kTI
        self.__dict__[FV.TI]  = ti

    def output_farm_vars(self, algo):
        return [FV.K]
    
    def calculate(self, algo, fdata, st_sel):

        kTI = self.get_data(FV.KTI, fdata, st_sel)
        ti  = self.get_data(FV.TI, fdata, st_sel)

        k = fdata.get(FV.K, 
                np.zeros((fdata.n_states, fdata.n_turbines), dtype=FC.DTYPE))
                
        k[st_sel] = kTI * ti

        return {FV.K: k}  
