import numpy as np
import pandas as pd

from foxes.core import States, VerticalProfile
from foxes.tools import PandasFileHelper
import foxes.variables as FV
import foxes.constants as FC

class StatesTable(States):

    RDICT = {'index_col': 0}

    def __init__(
        self,
        data_source,
        output_vars,
        var2col={},
        fixed_vars={},
        profiles={},
        pd_read_pars={},
        verbosity=1
    ):
        super().__init__()

        self._data      = data_source
        self.ovars      = output_vars
        self.rpars      = pd_read_pars
        self.var2col    = var2col
        self.fixed_vars = fixed_vars
        self.profdicts  = profiles

        if not isinstance(self._data, pd.DataFrame):
            if verbosity:
                print(f"States '{self.name}': Reading file {self._data}")
            rpars      = dict(self.RDICT, **self.rpars)
            self._data = PandasFileHelper().read_file(self._data, **rpars)
        self.N = len(self._data.index)

    def input_farm_data(self, algo):

        self.VARS = self.var("vars")
        self.DATA = self.var("data")

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        self._weights = np.zeros((self.N, algo.n_turbines), dtype=FC.DTYPE)
        if col_w in self._data:
            self._weights[:] = self._data[col_w].to_numpy()[:, None]
        elif FV.WEIGHT in self.var2col:
            raise KeyError(f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {self._data.columns}")
        else:
            self._weights[:] = 1./self.N

        self.profiles = {}
        self.tvars    = set(self.ovars)
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self.profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self.profiles[v] = d
            elif isinstance(d, dict):
                t = d.pop("type")
                self.profiles[v] = VerticalProfile.new(t, **d)
            else:
                raise TypeError(f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict")
            self.tvars.update(self.profiles[v].input_vars())
        self.tvars -= set(self.fixed_vars.keys())
        self.tvars  = list(self.tvars)

        tcols = []
        for v in self.tvars:
            c = self.var2col.get(v, v)
            if c in self._data.columns:
                tcols.append(c)
            elif v not in self.profiles.keys():
                raise KeyError(f"States '{self.name}': Missing variable '{c}' in states table columns, profiles or fixed vars")
        self._data = self._data[tcols]

        idata  = super().input_farm_data(algo)

        if self._data.index.name is not None:
            idata["coords"][FV.STATE] = self._data.index.to_numpy()
        
        idata["coords"][self.VARS]    = self.tvars
        idata["data_vars"][self.DATA] = ((FV.STATE, self.VARS), self._data.to_numpy())

        del self._data

        return idata

    def initialize(self, algo, farm_data, point_data):
        super().initialize(algo, farm_data, point_data)

        for p in self.profiles.values():
            if not p.initialized:
                p.initialize(algo, point_data)

    def size(self):
        return self.N

    def output_point_vars(self, algo):
        return self.ovars

    def weights(self, algo):
        return self._weights

    def calculate(self, algo, fdata, pdata):

        n_states = len(fdata[FV.STATE])
        n_points = pdata.n_points
        z        = pdata[FV.POINTS][:, :, 2]
        
        out = {}
        for i, v in enumerate(self.tvars):
            if v not in pdata:
                pdata[v] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            if v in self.ovars:
                out[v] = pdata[v]
            pdata[v][:] = fdata[self.DATA][:, i, None]
        
        for v, f in self.fixed_vars.items():
            if v not in pdata:
                pdata[v] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            if v in self.ovars:
                out[v] = pdata[v]
            pdata[v][:] = f 

        for v, p in self.profiles.items():
            if v not in pdata:
                pdata[v] = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            if v in self.ovars:
                out[v] = pdata[v]
            pres = p.calculate(pdata, z)
            pdata[v][:] = pres

        return out
