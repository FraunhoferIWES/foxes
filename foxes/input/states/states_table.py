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
        states_sel=None,
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
        if states_sel is not None:
            self._data = self._data.iloc[states_sel]
        self._N = len(self._data.index)

    def model_input_data(self, algo):

        self.VARS = self.var("vars")
        self.DATA = self.var("data")

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        self._weights = np.zeros((self._N, algo.n_turbines), dtype=FC.DTYPE)
        if col_w in self._data:
            self._weights[:] = self._data[col_w].to_numpy()[:, None]
        elif FV.WEIGHT in self.var2col:
            raise KeyError(f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {self._data.columns}")
        else:
            self._weights[:] = 1./self._N

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

        idata  = super().model_input_data(algo)

        if self._data.index.name is not None:
            idata["coords"][FV.STATE] = self._data.index.to_numpy()
        
        idata["coords"][self.VARS]    = self.tvars
        idata["data_vars"][self.DATA] = ((FV.STATE, self.VARS), self._data.to_numpy())

        del self._data

        return idata

    def initialize(self, algo):
        super().initialize(algo)

        for p in self.profiles.values():
            if not p.initialized:
                p.initialize(algo)

    def size(self):
        return self._N

    def output_point_vars(self, algo):
        return self.ovars

    def weights(self, algo):
        return self._weights

    def calculate(self, algo, mdata, fdata, pdata):

        z = pdata[FV.POINTS][:, :, 2]
        
        for i, v in enumerate(self.tvars):
            pdata[v][:] = mdata[self.DATA][:, i, None]
        
        for v, f in self.fixed_vars.items():
            pdata[v] = np.full((pdata.n_states, pdata.n_points), f, dtype=FC.DTYPE) 

        for v, p in self.profiles.items():
            pres = p.calculate(pdata, z)
            pdata[v] = pres
        
        return {v: pdata[v] for v in self.output_point_vars(algo)}
