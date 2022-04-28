from abc import ABCMeta, abstractmethod
import numpy as np
import xarray as xr

import foxes.variables as FV
from foxes.core.point_data_model import PointDataModel
from foxes.core.point_data_model_list import PointDataModelList

class Algorithm(metaclass=ABCMeta):

    def __init__(self, mbook, farm, chunks, verbosity):
        
        self.name         = type(self).__name__
        self.mbook        = mbook
        self.farm         = farm
        self.chunks       = chunks
        self.verbosity    = verbosity
        self.n_turbines   = farm.n_turbines
        self.models_idata = {"coords": {}, "data_vars": {}}
    
    def print(self, *args, **kwargs):
        if self.verbosity > 0:
            print(*args, **kwargs)

    @abstractmethod
    def calc_farm(self):
        pass

    def calc_points(
            self, 
            farm_data, 
            points, 
            vars=None, 
            point_models=None,
            init_pars=None,
            calc_pars=None,
            final_pars=None
        ):

        # update models:
        pmodels = [self.states]
        ipars   = [{}]
        cpars   = [{}]
        fpars   = [{}]
        if point_models is not None:
            if not isinstance(point_models, list):
                point_models = [point_models]
            for mi, m in enumerate(point_models):
                if isinstance(m, str):
                    pname  = m
                    pmodel = self.mbook.point_models[pname]
                    pmodel.name = pname
                    pmodels.append(pmodel)
                elif isinstance(m, PointDataModel):
                    pmodels.append(m)
                else:
                    raise TypeError(f"Model '{m}' is neither str nor PointDataModel")
                ipars.append({} if init_pars is None else init_pars[mi])
                cpars.append({} if calc_pars is None else calc_pars[mi])
                fpars.append({} if final_pars is None else final_pars[mi])

        # create model list:
        pmodels = PointDataModelList(pmodels)

        # create data, filled with zeros:
        point_data = self.new_point_data(pmodels.input_point_data(self), self.chunks, points).persist()
        self.print("\nInput point data:\n\n", point_data, "\n")

        # initialize models:
        pmodels.initialize(self, farm_data, point_data, parameters=ipars, verbosity=self.verbosity)
        
        # check vars:
        ovars = pmodels.output_point_vars(self)
        if vars is None:
            vars = ovars
        self.print(f"Calculating {len(vars)} variables with model '{pmodels.name}':")
        self.print(", ".join(ovars))
        for v in vars:
            if v not in ovars:
                raise KeyError(f"Variable '{v}' not in output point vars of model '{pmodels.name}': {ovars}")

        # calculate:
        pdata = pmodels.run_calculation(self, farm_data, point_data, vars, parameters=cpars)

        # finalize models:
        self.print("\n")
        pmodels.finalize(self, farm_data, point_data, parameters=fpars, verbosity=self.verbosity)

        return pdata

    def new_farm_data(self, input_data, chunks):

        idata = self.states.input_farm_data(self)
        idata["coords"].update(input_data["coords"])
        idata["data_vars"].update(input_data["data_vars"])

        sizes = {}
        for v, t in idata["data_vars"].items():
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError(f"Input farm data entry '{v}': Not a tuple of size 2, got '{t}'")
            if not isinstance(t[0], tuple):
                raise ValueError(f"Input farm data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            for c in t[0]:
                if not isinstance(c, str):
                    raise ValueError(f"Input farm data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            if not isinstance(t[1], np.ndarray):
                raise ValueError(f"Input farm data entry '{v}': Second entry is not a numpy array, got: {type(t[1]).__name__}")
            if len(t[1].shape) != len(t[0]):
                raise ValueError(f"Input farm data entry '{v}': Wrong data shape, expecting {len(t[0])} dimensions, got {t[1].shape}")
            if FV.STATE in t[0]:
                if t[0][0] != FV.STATE:
                    raise ValueError(f"Input farm data entry '{v}': Dimension '{FV.STATE}' not at first position, got {t[0]}")
            for d, s in zip(t[0], t[1].shape):
                if d not in sizes:
                    sizes[d] = s
                elif sizes[d] != s:
                    raise ValueError(f"Input farm data entry '{v}': Dimension '{d}' has wrong size, expecting {sizes[d]}, got {s}")

        for v, c in idata["coords"].items():
            if v not in sizes:
                raise KeyError(f"Input coords entry '{v}': Not used in farm data, found {sorted(list(sizes.keys()))}")
            elif len(c) != sizes[v]:
                raise ValueError(f"Input coords entry '{v}': Wrong coordinate size for '{v}': Expecting {sizes[v]}, got {len(c)}")

        farm_data = xr.Dataset(**idata)
        
        if chunks is not None:
            if FV.TURBINE in chunks.keys():
                raise ValueError(f"Dimension '{FV.TURBINE}' cannot be chunked, got chunks {chunks}")
            if FV.RPOINT in chunks.keys():
                raise ValueError(f"Dimension '{FV.RPOINT}' cannot be chunked, got chunks {chunks}")
            farm_data = farm_data.chunk(chunks={c: v for c, v in chunks.items() if c in sizes})
        
        return farm_data

    def new_point_data(self, input_data, chunks, points):

        idata = {"coords": {}, "data_vars": {}}
        idata["coords"].update(input_data["coords"])
        idata["data_vars"].update(input_data["data_vars"])
        
        if len(points.shape) != 3 or points.shape[2] != 3:
            raise ValueError(f"points have wrong dimensions, expecting (n_states, n_points, 3), got {points.shape}")
        idata["data_vars"][FV.POINTS] = ((FV.STATE, FV.POINT, FV.XYH), points)

        sizes = {}
        for v, t in idata["data_vars"].items():
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError(f"Input point data entry '{v}': Not a tuple of size 2, got '{t}'")
            if not isinstance(t[0], tuple):
                raise ValueError(f"Input point data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            for c in t[0]:
                if not isinstance(c, str):
                    raise ValueError(f"Input point data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'")
            if not isinstance(t[1], np.ndarray):
                raise ValueError(f"Input point data entry '{v}': Second entry is not a numpy array, got: {type(t[1]).__name__}")
            if len(t[1].shape) != len(t[0]):
                raise ValueError(f"Input point data entry '{v}': Wrong data shape, expecting {len(t[0])} dimensions, got {t[1].shape}")
            if FV.STATE in t[0]:
                if t[0][0] != FV.STATE:
                    raise ValueError(f"Input point data entry '{v}': Dimension '{FV.STATE}' not at first position, got {t[0]}")
                if FV.POINT in t[0] and t[0][1] != FV.POINT:
                    raise ValueError(f"Input point data entry '{v}': Dimension '{FV.POINT}' not at second position, got {t[0]}")
            elif FV.POINT in t[0]:
                if t[0][0] != FV.POINT:
                    raise ValueError(f"Input point data entry '{v}': Dimension '{FV.POINT}' not at first position, got {t[0]}")
            for d, s in zip(t[0], t[1].shape):
                if d not in sizes:
                    sizes[d] = s
                elif sizes[d] != s:
                    raise ValueError(f"Input point data entry '{v}': Dimension '{d}' has wrong size, expecting {sizes[d]}, got {s}")

        for v, c in idata["coords"].items():
            if v not in sizes:
                raise KeyError(f"Input coords entry '{v}': Not used in point data, found {sorted(list(sizes.keys()))}")
            elif len(c) != sizes[v]:
                raise ValueError(f"Input coords entry '{v}': Wrong coordinate size for '{v}': Expecting {sizes[v]}, got {len(c)}")

        point_data = xr.Dataset(**idata)
        
        if chunks is not None:
            if FV.TURBINE in chunks.keys():
                raise ValueError(f"Dimension '{FV.TURBINE}' cannot be chunked, got chunks {chunks}")
            point_data = point_data.chunk(chunks={c: v for c, v in chunks.items() if c in sizes})
        
        return point_data