import numpy as np
import xarray as xr

from .model import Model
from foxes.data import StaticData
from foxes.utils import Dict, all_subclasses
import foxes.constants as FC


class Algorithm(Model):
    """
    Abstract base class for algorithms.

    Algorithms collect required objects for running
    calculations, and contain the calculation functions
    which are meant to be called from top level code.

    Attributes
    ----------
    mbook: foxes.models.ModelBook
        The model book
    farm: foxes.WindFarm
        The wind farm
    chunks: dict
        The chunks choice for running in parallel with dask,
        e.g. `{"state": 1000}` for chunks of 1000 states
    verbosity: int
        The verbosity level, 0 means silent
    dbook: foxes.DataBook
        The data book, or None for default

    :group: core

    """

    def __init__(self, mbook, farm, chunks, verbosity, dbook=None):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.models.ModelBook
            The model book
        farm: foxes.WindFarm
            The wind farm
        chunks: dict
            The chunks choice for running in parallel with dask,
            e.g. `{"state": 1000}` for chunks of 1000 states
        verbosity: int
            The verbosity level, 0 means silent
        dbook: foxes.DataBook, optional
            The data book, or None for default

        """
        super().__init__()

        self.name = type(self).__name__
        self.mbook = mbook
        self.farm = farm
        self.chunks = chunks
        self.verbosity = verbosity
        self.n_states = None
        self.n_turbines = farm.n_turbines
        self.dbook = StaticData() if dbook is None else dbook

        if chunks is not None and FC.TARGET not in chunks:
            self.chunks[FC.TARGET] = chunks.get(FC.POINT, None)

        self._idata_mem = Dict()

    def print(self, *args, vlim=1, **kwargs):
        """
        Print function, based on verbosity.

        Parameters
        ----------
        args: tuple, optional
            Arguments for the print function
        kwargs: dict, optional
            Keyword arguments for the print function
        vlim: int
            The verbosity limit

        """
        if self.verbosity >= vlim:
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
            if FC.STATE in t[0]:
                if t[0][0] != FC.STATE:
                    raise ValueError(
                        f"Input {mtype} data entry '{v}': Dimension '{FC.STATE}' not at first position, got {t[0]}"
                    )
                if FC.TURBINE in t[0]:
                    if t[0][1] != FC.TURBINE:
                        raise ValueError(
                            f"Input {mtype} data entry '{v}': Dimension '{FC.TURBINE}' not at second position, got {t[0]}"
                        )
                if FC.TARGET in t[0]:
                    if t[0][1] != FC.TARGET:
                        raise ValueError(
                            f"Input {mtype} data entry '{v}': Dimension '{FC.TARGET}' not at second position, got {t[0]}"
                        )
                    if len(t[0]) < 3 or t[0][2] != FC.TPOINT:
                        raise KeyError(
                            f"Input {mtype} data entry '{v}': Expecting dimension '{FC.TPOINT}' as third entry. Got {t[0]}"
                        )
            elif FC.TURBINE in t[0]:
                raise ValueError(
                    f"Input {mtype} data entry '{v}': Dimension '{FC.TURBINE}' requires combination with dimension '{FC.STATE}'"
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
            if FC.TURBINE in self.chunks.keys():
                raise ValueError(
                    f"Dimension '{FC.TURBINE}' cannot be chunked, got chunks {self.chunks}"
                )
            if FC.TPOINT in self.chunks.keys():
                raise ValueError(
                    f"Dimension '{FC.TPOINT}' cannot be chunked, got chunks {self.chunks}"
                )
            xrdata = xrdata.chunk(
                chunks={c: v for c, v in self.chunks.items() if c in sizes}
            )
        return xrdata

    def chunked(self, ds):
        return (
            ds.chunk(chunks={c: v for c, v in self.chunks.items() if c in ds.coords})
            if self.chunks is not None
            else ds
        )

    def initialize(self):
        """
        Initializes the algorithm.
        """
        super().initialize(self, self.verbosity)

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

    def store_model_data(self, model, idata, force=False):
        """
        Store model data

        Parameters
        ----------
        model: foxes.core.Model
            The model
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`
        force: bool
            Overwrite existing data

        """
        mname = f"{type(model).__name__}_{model.name}"
        if not force and mname in self._idata_mem:
            raise KeyError(f"Attempt to overwrite stored data for model '{mname}'")
        self._idata_mem[mname] = idata

    def get_model_data(self, model):
        """
        Gets model data from memory

        Parameters
        ----------
        model: foxes.core.Model
            The model

        """
        mname = f"{type(model).__name__}_{model.name}"
        try:
            return self._idata_mem[mname]
        except KeyError:
            raise KeyError(
                f"Key '{mname}' not found in idata_mem, available keys: {sorted(list(self._idata_mem.keys()))}"
            )

    def del_model_data(self, model):
        """
        Remove stored model data

        Parameters
        ----------
        model: foxes.core.Model
            The model

        """
        mname = f"{type(model).__name__}_{model.name}"
        try:
            del self._idata_mem[mname]
        except KeyError:
            raise KeyError(f"Attempt to delete data of model '{mname}', but not stored")

    def update_n_turbines(self):
        """
        Reset the number of turbines,
        according to self.farm
        """
        if self.n_turbines != self.farm.n_turbines:
            self.n_turbines = self.farm.n_turbines

            # resize stored idata, if dependent on turbine coord:
            newk = {}
            for mname, idata in self.idata_mem.items():
                if mname[:2] == "__":
                    continue
                for dname, d in idata["data_vars"].items():
                    k = f"__{mname}_{dname}_turbinv"
                    if k in self.idata_mem:
                        ok = self.idata_mem[k]
                    else:
                        ok = None
                        if FC.TURBINE in d[0]:
                            i = d[0].index(FC.TURBINE)
                            ok = np.unique(d[1], axis=1).shape[i] == 1
                        newk[k] = ok
                    if ok is not None:
                        if not ok:
                            raise ValueError(
                                f"{self.name}: Stored idata entry '{mname}:{dname}' is turbine dependent, unable to reset n_turbines"
                            )
                        if FC.TURBINE in idata["coords"]:
                            idata["coords"][FC.TURBINE] = np.arange(self.n_turbines)
                        i = d[0].index(FC.TURBINE)
                        n0 = d[1].shape[i]
                        if n0 > self.n_turbines:
                            idata["data_vars"][dname] = (
                                d[0],
                                np.take(d[1], range(self.n_turbines), axis=i),
                            )
                        elif n0 < self.n_turbines:
                            shp = [
                                d[1].shape[j] if j != i else self.n_turbines - n0
                                for j in range(len(d[1].shape))
                            ]
                            a = np.zeros(shp, dtype=d[1].dtype)
                            shp = [
                                d[1].shape[j] if j != i else 1
                                for j in range(len(d[1].shape))
                            ]
                            a[:] = np.take(d[1], -1, axis=i).reshape(shp)
                            idata["data_vars"][dname] = (
                                d[0],
                                np.append(d[1], a, axis=i),
                            )

            self._idata_mem.update(newk)

    def get_models_idata(self):
        """
        Returns idata object of models

        Returns
        -------
        idata: dict, optional
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`.
            Take algorithm's idata object by default.

        """
        if not self.initialized:
            raise ValueError(
                f"Algorithm '{self.name}': get_models_idata called before initialization"
            )
        idata = {"coords": {}, "data_vars": {}}
        for k, hidata in self._idata_mem.items():
            if len(k) < 3 or k[:2] != "__":
                idata["coords"].update(hidata["coords"])
                idata["data_vars"].update(hidata["data_vars"])
        return idata

    def get_models_data(self, idata=None):
        """
        Creates xarray from model input data.

        Parameters
        ----------
        idata: dict, optional
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
            idata = self.get_models_idata()
        sizes = self.__get_sizes(idata, "models")
        return self.__get_xrdata(idata, sizes)

    def new_point_data(self, points, states_indices=None):
        """
        Creates a point data xarray object, containing only points.

        Parameters
        ----------
        points: numpy.ndarray
            The points, shape: (n_states, n_points, 3)
        states_indices: array_like, optional
            The indices of the states dimension

        Returns
        -------
        xarray.Dataset
            A dataset containing the points data

        """

        if states_indices is None:
            idata = {"coords": {}, "data_vars": {}}
        else:
            idata = {"coords": {FC.STATE: states_indices}, "data_vars": {}}

        if (
            len(points.shape) != 3
            or points.shape[0] != self.n_states
            or points.shape[2] != 3
        ):
            raise ValueError(
                f"points have wrong dimensions, expecting ({self.n_states}, {points.shape[1]}, 3), got {points.shape}"
            )
        idata["data_vars"][FC.TARGETS] = (
            (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH),
            points[:, :, None, :],
        )
        idata["data_vars"][FC.TWEIGHTS] = (
            (FC.TPOINT,),
            np.array([1.0], dtype=FC.DTYPE),
        )

        sizes = self.__get_sizes(idata, "point")
        return self.__get_xrdata(idata, sizes)

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem: bool
            Clear idata memory

        """
        super().finalize(self, self.verbosity)
        if clear_mem:
            self._idata_mem = Dict()

    @classmethod
    def new(cls, algo_type, *args, **kwargs):
        """
        Run-time algorithm factory.

        Parameters
        ----------
        algo_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """

        if algo_type is None:
            return None

        allc = all_subclasses(cls)
        found = algo_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == algo_type:
                    return scls(*args, **kwargs)

        else:
            estr = (
                "Algorithm type '{}' is not defined, available types are \n {}".format(
                    algo_type, sorted([i.__name__ for i in allc])
                )
            )
            raise KeyError(estr)
