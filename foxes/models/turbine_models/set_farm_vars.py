import numpy as np

from foxes.core import TurbineModel
import foxes.constants as FC
import foxes.variables as FV

class SetFarmVars(TurbineModel):

    def __init__(self):
        super().__init__()

        self.vars   = []
        self._vdata = []
    
    def add_var(self, var, data):
        self.vars.append(var)
        self._vdata.append(data)
    
    def output_farm_vars(self, algo):
        return self.vars
    
    def model_input_data(self, algo):
        
        idata = super().model_input_data(algo)

        self._keys = {}
        for i, v in enumerate(self.vars):

            if not isinstance(self._vdata[i], np.ndarray):
                raise TypeError(f"Model '{self.name}': Wrong data type for variable '{v}': Expecting '{np.ndarray.__name__}', got '{type(self._vdata[i]).__name__}'")

            data = np.full((algo.n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE)
            data[:] = self._vdata[i]

            k = self.var(f"data_{v}")
            self._keys[v] = k

            idata["data_vars"][k] = ((FV.STATE, FV.TURBINE), data)
        
        del self._vdata

        return idata
    
    def calculate(self, algo, mdata, fdata, st_sel):

        n_states   = fdata.n_states
        n_turbines = fdata.n_turbines
        allt       = np.all(st_sel)       

        for v in self.vars:

            data  = mdata[self._keys[v]]
            hsel  = ~np.isnan(data)
            hallt = np.all(hsel)

            if allt and hallt:
                fdata[v] = data
                
            else:

                if v not in fdata:
                    fdata[v] = np.full((n_states, n_turbines), np.nan, dtype=FC.DTYPE)

                tsel = st_sel & hsel
                fdata[v][tsel] = data[tsel]

        return {v: fdata[v] for v in self.vars}
