import numpy as np
import xarray as xr

from .model import Model
from foxes.data import StaticData
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


    """

    def __init__(self, mbook, farm, chunks, verbosity, dbook=None):
        super().__init__()

        self.name = type(self).__name__
        self.mbook = mbook
        self.farm = farm
        self.chunks = chunks
        self.verbosity = verbosity
        self.n_states = None
        self.n_turbines = farm.n_turbines
        self.dbook = StaticData() if dbook is None else dbook

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
        super().initialize(self, verbosity=self.verbosity)

    def model_input_data(self, algo):
        """
        The algorithm input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        raise NotImplementedError(
            f"Algorithm '{self.name}': model_input_data called, illegally"
        )

    def get_models_data(self, models):
        """
        Creates xarray from model input data.

        Parameters
        ----------
        models : array_like of foxes.core.Model
            The models whose data to collect

        Returns
        -------
        xarray.Dataset
            The model input data

        """
        if not isinstance(models, tuple) and not isinstance(models, list):
            models = [models]

        idata = {"coords": {}, "data_vars": {}}
        for m in models:
            hidata = m.model_input_data(self)
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

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem : bool
            Flag for deleting algorithm data and
            resetting initialization flag

        """
        super().finalize(self, clear_mem=clear_mem, verbosity=self.verbosity)
