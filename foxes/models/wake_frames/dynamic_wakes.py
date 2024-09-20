import numpy as np
from xarray import Dataset

from foxes.core import WakeFrame, MData, FData, TData
from foxes.utils import wd2uv
from foxes.algorithms.iterative import Iterative
import foxes.variables as FV
import foxes.constants as FC

class DynamicWakes(WakeFrame):
    """
    Dynamic wakes for any kind of timeseries states.

    Attributes
    ----------
    max_length_km: float
        The maximal wake length in km
    max_age: int
        The maximal number of wake steps
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation
    dt_min: float
        The delta t value in minutes,
        if not from timeseries data

    :group: models.wake_frames

    """
    def __init__(self, max_length_km=20, max_age=None, cl_ipars={}, dt_min=None):
        """
        Constructor.

        Parameters
        ----------
        max_length_km: float
            The maximal wake length in km
        max_age: int, optional
            The maximal number of wake steps
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data

        """
        super().__init__()

        self.max_length_km = max_length_km
        self.max_age = max_age
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min

    def __repr__(self):
        return f"{type(self).__name__}(dt_min={self.dt_min}, max_length_km={self.max_length_km}, max_age={self.max_age})"

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        # get and check times:
        times = np.asarray(algo.states.index())
        if self.dt_min is None:
            if not np.issubdtype(times.dtype, np.datetime64):
                raise TypeError(
                    f"{self.name}: Expecting state index of type np.datetime64, found {times.dtype}"
                )
            elif len(times) == 1:
                raise KeyError(
                    f"{self.name}: Expecting 'dt_min' for single step timeseries"
                )
            dt = (times[1:] - times[:-1]).astype("timedelta64[s]").astype(FC.ITYPE)
        else:
            n = max(len(times) - 1, 1)
            dt = np.full(n, self.dt_min * 60, dtype="timedelta64[s]").astype(FC.ITYPE)
        dt = np.append(dt, dt[-1], axis=0)
        
        # find max age if not given:
        if self.max_age is None:
            dist = np.sum(20*self._dt)
            self.max_age = max(int(dist/self.max_length_km/1e3), 1)
            if verbosity > 0:
                print(f"{self.name}: Setting max_age = {self.max_age}")
        
        # init data:
        TODO - is this a good idea?
        idata = super().load_data(algo, verbosity=verbosity)
        self.DATA = self.var("data")
        self.AGE = self.var("age")
        self.DT = self.var("dt")
        data = np.full(
            (algo.n_states, algo.n_turbines, self.max_age, 3), # x, y, length
            np.nan, 
            dtype=FC.DTYPE,
        )
        idata.data_vars[self.DATA] = ((FC.STATE, FC.TURBINE, self.AGE), data)
        idata.data_vars[self.DT] = ((FC.STATE,), dt)
        
        return idata
            
    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not isinstance(algo, Iterative):
            raise TypeError(
                f"Incompatible algorithm type {type(algo).__name__}, expecting {Iterative.__name__}"
            )
        super().initialize(algo, verbosity)
        
    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        order = np.zeros((fdata.n_states, fdata.n_turbines), dtype=FC.ITYPE)
        order[:] = np.arange(fdata.n_turbines)[None, :]
        return order
    
    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
        """
        Calculate wake coordinates of rotor points.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        # prepare:
        targets = tdata[FC.TARGETS]
        n_states, n_targets, n_tpoints = targets.shape[:3]
        n_points = n_targets * n_tpoints
        rxyh = fdata[FV.TXYH][:, downwind_index]
        
        

            