import numpy as np
import xarray as xr

from .model import Model
from .farm_data_model import FarmDataModelList
from .point_data_model import PointDataModelList
from .farm_controller import FarmController
from foxes.data import StaticData
from foxes.utils import Dict
import foxes.variables as FV


class Algorithm(Model):
    """
    Abstract base class for algorithms.

    Algorithms collect required objects for running
    calculations, and contain the calculation functions
    which are meant to be called from top level code.

    Parameters
    ----------
    mbook : foxes.ModelBook
        The model book
    farm : foxes.WindFarm
        The wind farm
    chunks : dict
        The chunks choice for running in parallel with dask,
        e.g. `{"state": 1000}` for chunks of 1000 states
    verbosity : int
        The verbosity level, 0 means silent
    dbook : foxes.DataBook, optional
        The data book, or None for default
    keep_models : list of str
        Keep these models data in memory and do not finalize them

    Parameters
    ----------
    mbook : foxes.ModelBook
        The model book
    farm : foxes.WindFarm
        The wind farm
    chunks : dict
        The chunks choice for running in parallel with dask,
        e.g. `{"state": 1000}` for chunks of 1000 states
    verbosity : int
        The verbosity level, 0 means silent
    dbook : foxes.DataBook
        The data book, or None for default
    keep_models : list of str
        Keep these models data in memory and do not finalize them

    """

    def __init__(self, mbook, farm, chunks, verbosity, dbook=None, keep_models=[]):
        super().__init__()

        self.name = type(self).__name__
        self.mbook = mbook
        self.farm = farm
        self.chunks = chunks
        self.verbosity = verbosity
        self.n_states = None
        self.n_turbines = farm.n_turbines
        self.dbook = StaticData() if dbook is None else dbook
        self.keep_models = keep_models

        self._idata_mem = Dict()

    def print(self, *args, **kwargs):
        """
        Print function, based on verbosity.
        """
        if self.verbosity > 0:
            print(*args, **kwargs)

    def __get_sizes(self, idata, mtype):
        """
        Private helper function
        """

        sizes = {}
        for v, t in idata["data_vars"].items():
            if not isinstance(t, tuple) or len(t) != 2:
                raise ValueError(
                    f"Input {mtype} data entry '{v}': Not a tuple of size 2, got '{t}'"
                )
            if not isinstance(t[0], tuple):
                raise ValueError(
                    f"Input {mtype} data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'"
                )
            for c in t[0]:
                if not isinstance(c, str):
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': First tuple entry not a dimensions tuple, got '{t[0]}'"
                    )
            if not isinstance(t[1], np.ndarray):
                raise ValueError(
                    f"Input {mtype} data entry '{v}': Second entry is not a numpy array, got: {type(t[1]).__name__}"
                )
            if len(t[1].shape) != len(t[0]):
                raise ValueError(
                    f"Input {mtype} data entry '{v}': Wrong data shape, expecting {len(t[0])} dimensions, got {t[1].shape}"
                )
            if FV.STATE in t[0]:
                if t[0][0] != FV.STATE:
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': Dimension '{FV.STATE}' not at first position, got {t[0]}"
                    )
                if FV.POINT in t[0] and t[0][1] != FV.POINT:
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': Dimension '{FV.POINT}' not at second position, got {t[0]}"
                    )
            elif FV.POINT in t[0]:
                if t[0][0] != FV.POINT:
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': Dimension '{FV.POINT}' not at first position, got {t[0]}"
                    )
            for d, s in zip(t[0], t[1].shape):
                if d not in sizes:
                    sizes[d] = s
                elif sizes[d] != s:
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': Dimension '{d}' has wrong size, expecting {sizes[d]}, got {s}"
                    )
        for v, c in idata["coords"].items():
            if v not in sizes:
                raise KeyError(
                    f"Input coords entry '{v}': Not used in farm data, found {sorted(list(sizes.keys()))}"
                )
            elif len(c) != sizes[v]:
                raise ValueError(
                    f"Input coords entry '{v}': Wrong coordinate size for '{v}': Expecting {sizes[v]}, got {len(c)}"
                )

        return sizes

    def __get_xrdata(self, idata, sizes):
        """
        Private helper function
        """
        xrdata = xr.Dataset(**idata)
        if self.chunks is not None:
            if FV.TURBINE in self.chunks.keys():
                raise ValueError(
                    f"Dimension '{FV.TURBINE}' cannot be chunked, got chunks {self.chunks}"
                )
            if FV.RPOINT in self.chunks.keys():
                raise ValueError(
                    f"Dimension '{FV.RPOINT}' cannot be chunked, got chunks {self.chunks}"
                )
            xrdata = xrdata.chunk(
                chunks={c: v for c, v in self.chunks.items() if c in sizes}
            )
        return xrdata

    def initialize(self):
        """
        Initializes the algorithm.
        """
        self._idata_mem[self.name] = super().initialize(self, self.verbosity)

    def update_idata(self, models, idata=None, verbosity=None):
        """
        Add to idata memory, optionally update and return idata object.

        Parameters
        ----------
        models : foxes.core.Model or list of foxes.core.Model
            The models to initialize
        idata : dict, optional
            The idata dictionary to be updated, else only add
            to idata memory
        verbosity : int, optional
            The verbosity level, 0 = silent
            
        """
        if idata is None and not self.initialized:
            raise ValueError(f"Algorithm '{self.name}': update_idata called before initialization")
        
        verbosity = self.verbosity if verbosity is None else verbosity

        if not isinstance(models, list) and not isinstance(models, tuple):
            models = [models]

        for m in models:

            if m.initialized:
                try:
                    hidata = self._idata_mem[m.name]
                except KeyError:
                    raise KeyError(f"Model '{m.name}' initialized but not found in idata memory")

            else:
                self.print(f"Initializing model '{m.name}'")
                hidata = m.initialize(self, verbosity)
                self._idata_mem[m.name] = hidata

            if idata is not None:
                idata["coords"].update(hidata["coords"])
                idata["data_vars"].update(hidata["data_vars"])

    @property
    def idata_mem(self):
        """
        The current idata memory

        Returns
        -------
        dict :
            Keys: model name, value: idata dict

        """
        return self._idata_mem

    def get_models_data(self, idata=None):
        """
        Creates xarray from model input data.

        Parameters
        ----------
        idata : dict, optional
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`.
            Take algorithm's idata object by default.

        Returns
        -------
        xarray.Dataset
            The model input data

        """
        if idata is None:
            if not self.initialized:
                raise ValueError(f"Algorithm '{self.name}': get_models_data called before initialization")
            idata = self._idata_mem.pop(self.name)
            mnames = list(self._idata_mem.keys())
            for mname in mnames:
                if mname in self.keep_models:
                    hidata = self._idata_mem.get(mname)
                else:
                    hidata = self._idata_mem.pop(mname)
                idata["coords"].update(hidata["coords"])
                idata["data_vars"].update(hidata["data_vars"])

        sizes = self.__get_sizes(idata, "models")
        return self.__get_xrdata(idata, sizes)

    def new_point_data(self, points, states_indices=None):
        """
        Creates a point data xarray object, containing only points.

        Parameters
        ----------
        points : numpy.ndarray
            The points, shape: (n_states, n_points, 3)
        states_indices : array_like, optional
            The indices of the states dimension

        Returns
        -------
        xarray.Dataset
            A dataset containing the points data

        """

        if states_indices is None:
            idata = {"coords": {}, "data_vars": {}}
        else:
            idata = {"coords": {FV.STATE: states_indices}, "data_vars": {}}

        if (
            len(points.shape) != 3
            or points.shape[0] != self.n_states
            or points.shape[2] != 3
        ):
            raise ValueError(
                f"points have wrong dimensions, expecting ({self.n_states}, n_points, 3), got {points.shape}"
            )
        idata["data_vars"][FV.POINTS] = ((FV.STATE, FV.POINT, FV.XYH), points)

        sizes = self.__get_sizes(idata, "point")
        return self.__get_xrdata(idata, sizes)

    def finalize_model(self, model, verbosity=None):
        """
        Call the finalization routine of the model,
        if not to be kept.

        Parameters
        ----------
        model : foxes.core.Model
            The model to be finalized, if not in the
            keep_models list
        verbosity : int, optional
            The verbosity level, 0 = silent

        """
        verbosity = self.verbosity if verbosity is None else verbosity
  
        pr = False
        if isinstance(model, FarmController):
            if verbosity > 1:
                print(f"Finalizing model '{model.name}'")
                print(f"-- {model.name}: Starting finalization -- ")
                pr = True
            self.finalize_model(model.pre_rotor_models, verbosity)
            self.finalize_model(model.post_rotor_models, verbosity)
        elif isinstance(model, FarmDataModelList) or isinstance(model, PointDataModelList):
            if verbosity > 1:
                print(f"Finalizing model '{model.name}'")
                print(f"-- {model.name}: Starting finalization -- ")
                pr = True
            for m in model.models:
                self.finalize_model(m, verbosity)

        if model.initialized and model.name not in self.keep_models:
            if not pr and verbosity > 0:
                print(f"Finalizing model '{model.name}'")
            model.finalize(self, verbosity)
            if model.name in self._idata_mem:
                del self._idata_mem[model.name]
        
        if pr:
            print(f"-- {model.name}: Finished finalization -- ")

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem : bool
            Clear idata memory, including keep_models entries

        """
        if clear_mem:
            self._idata_mem = Dict()
        if self.initialized:
            super().finalize(self, self.verbosity)
