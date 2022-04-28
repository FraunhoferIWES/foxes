import numpy as np

from foxes.core import TurbineType
from foxes.tools import PandasFileHelper
import foxes.variables as FV

class PCtFile(TurbineType):

    def __init__(
        self,
        filepath,
        col_ws="ws",
        col_P="P",
        col_ct="ct",
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars={},
        **parameters
    ):
        super().__init__(**parameters)

        self.fpath  = filepath
        self.col_ws = col_ws
        self.col_P  = col_P
        self.col_ct = col_ct
        self.WSCT   = var_ws_ct
        self.WSP    = var_ws_P
        self.rpars  = pd_file_read_pars

    def output_farm_vars(self, algo):
        return [FV.P, FV.CT]
    
    def initialize(self, algo, farm_data, st_sel):
        data = PandasFileHelper.read_file(self.fpath, **self.rpars)
        data = data.set_index(self.col_ws).sort_index()
        self.data_ws = data.index.to_numpy()
        self.data_P  = data[self.col_P].to_numpy()
        self.data_ct = data[self.col_ct].to_numpy()
        super().initialize(algo, farm_data, st_sel)
    
    def calculate(self, algo, fdata, st_sel):
        
        rews2 = fdata[self.WSCT][st_sel]
        rews3 = fdata[self.WSP][st_sel] if self.WSP != self.WSCT else rews2

        out = {
            FV.P : fdata.get(FV.P, np.zeros_like(fdata[self.WSCT])),
            FV.CT: fdata.get(FV.CT, np.zeros_like(fdata[self.WSP]))
        }

        out[FV.P][st_sel]  = np.interp(rews3, self.data_ws, self.data_P, left=0., right=0.)
        out[FV.CT][st_sel] = np.interp(rews2, self.data_ws, self.data_ct, left=0., right=0.)

        return out
        