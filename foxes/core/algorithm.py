from abc import ABCMeta, abstractmethod
import numpy as np
import xarray as xr

import foxes.variables as FV

class Algorithm(metaclass=ABCMeta):

    def __init__(self, mbook, farm, chunks, verbosity):
        
        self.name         = type(self).__name__
        self.mbook        = mbook
        self.farm         = farm
        self.chunks       = chunks
        self.verbosity    = verbosity
        self.n_states     = None
        self.n_turbines   = farm.n_turbines
        self.models_idata = {}
    
    def print(self, *args, **kwargs):
        if self.verbosity > 0:
            print(*args, **kwargs)

    @abstractmethod
    def calc_farm(self):
        pass

    @abstractmethod
    def calc_points(
            self, 
            farm_data, 
            points, 
            vars=None, 
            point_models=None,
            init_pars={},
            calc_pars={},
            final_pars={}
        ):
        pass

    def __get_sizes(self, idata, mtype):

        sizes = {}
        for v, t in idata["data_vars"].items():
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError(f"Input {mtype} data entry '{v}': Not a tuple of size 2, got '{t}'")
            if not isinstance(t[0], tuple):
                raise ValueError(f"Input {mtype} data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            for c in t[0]:
                if not isinstance(c, str):
                    raise ValueError(f"Input {mtype} data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            if not isinstance(t[1], np.ndarray):
                raise ValueError(f"Input {mtype} data entry '{v}': Second entry is not a numpy array, got: {type(t[1]).__name__}")
            if len(t[1].shape) != len(t[0]):
                raise ValueError(f"Input {mtype} data entry '{v}': Wrong data shape, expecting {len(t[0])} dimensions, got {t[1].shape}")
            if FV.STATE in t[0]:
                if t[0][0] != FV.STATE:
                    raise ValueError(f"Input {mtype} data entry '{v}': Dimension '{FV.STATE}' not at first position, got {t[0]}")
                if FV.POINT in t[0] and t[0][1] != FV.POINT:
                    raise ValueError(f"Input {mtype} data entry '{v}': Dimension '{FV.POINT}' not at second position, got {t[0]}")
            elif FV.POINT in t[0]:
                if t[0][0] != FV.POINT:
                    raise ValueError(f"Input {mtype} data entry '{v}': Dimension '{FV.POINT}' not at first position, got {t[0]}")
            for d, s in zip(t[0], t[1].shape):
                if d not in sizes:
                    sizes[d] = s
                elif sizes[d] != s:
                    raise ValueError(f"Input {mtype} data entry '{v}': Dimension '{d}' has wrong size, expecting {sizes[d]}, got {s}")

        for v, c in idata["coords"].items():
            if v not in sizes:
                raise KeyError(f"Input coords entry '{v}': Not used in farm data, found {sorted(list(sizes.keys()))}")
            elif len(c) != sizes[v]:
                raise ValueError(f"Input coords entry '{v}': Wrong coordinate size for '{v}': Expecting {sizes[v]}, got {len(c)}")

        return sizes
    
    def __get_xrdata(self, idata, sizes):
        xrdata = xr.Dataset(**idata)
        if self.chunks is not None:
            if FV.TURBINE in self.chunks.keys():
                raise ValueError(f"Dimension '{FV.TURBINE}' cannot be chunked, got chunks {self.chunks}")
            if FV.RPOINT in self.chunks.keys():
                raise ValueError(f"Dimension '{FV.RPOINT}' cannot be chunked, got chunks {self.chunks}")
            xrdata = xrdata.chunk(chunks={c: v for c, v in self.chunks.items() if c in sizes})
        return xrdata

    def get_models_data(self):

        idata = {"coords": {}, "data_vars": {}}
        for ida in self.models_idata.values():
            idata["coords"].update(ida["coords"])
            idata["data_vars"].update(ida["data_vars"])

        sizes = self.__get_sizes(idata, "models")
        return self.__get_xrdata(idata, sizes)

    def new_point_data(self, points):
        
        idata = {"coords": {}, "data_vars": {}}
        if len(points.shape) != 3 or points.shape[0] != self.n_states or points.shape[2] != 3:
            raise ValueError(f"points have wrong dimensions, expecting ({self.n_states}, n_points, 3), got {points.shape}")
        idata["data_vars"][FV.POINTS] = ((FV.STATE, FV.POINT, FV.XYH), points)

        sizes = self.__get_sizes(idata, "point")
        return self.__get_xrdata(idata, sizes)
        