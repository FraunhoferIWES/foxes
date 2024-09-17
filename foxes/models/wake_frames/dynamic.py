import numpy as np
from xarray import Dataset

from foxes.core import WakeFrame, MData, FData, TData
from foxes.utils import wd2uv
import foxes.variables as FV
import foxes.constants as FC

class DynamicWakes(WakeFrame):
    """
    Dynamic wakes for any kind of timeseries states.

    Attributes
    ----------
    max_wake_length: float
        The maximal wake length
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation
    dt_min: float, optional
        The delta t value in minutes,
        if not from timeseries data
    heights: np.ndarray
        The turbine hub heights, shape: (n_turbines,)

    :group: models.wake_frames

    """
    def __init__(self, max_wake_length=2e4, cl_ipars={}, dt_min=None):
        """
        Constructor.

        Parameters
        ----------
        max_wake_length: float
            The maximal wake length
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data

        """
        super().__init__()

        self.max_wake_length = max_wake_length
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min

    def __repr__(self):
        return f"{type(self).__name__}(dt_min={self.dt_min})"
    
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
        super().initialize(algo, verbosity)

        # find turbine hub heights:
        t2h = np.zeros(algo.n_turbines, dtype=FC.DTYPE)
        for ti, t in enumerate(algo.farm.turbines):
            t2h[ti] = (
                t.H if t.H is not None else algo.farm_controller.turbine_types[ti].H
            )
        self.heights = np.unique(t2h)
        
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
        # prepare:
        n_states = fdata.n_states
        n_turbines = algo.n_turbines
        tdata = TData.from_points(points=fdata[FV.TXYH])

        # calculate streamline x coordinates for turbines rotor centre points:
        # n_states, n_turbines_source, n_turbines_target
        coosx = np.zeros((n_states, n_turbines, n_turbines), dtype=FC.DTYPE)
        for ti in range(n_turbines):
            coosx[:, ti, :] = self.get_wake_coos(algo, mdata, fdata, tdata, ti)[
                :, :, 0, 0
            ]

        # derive turbine order:
        # TODO: Remove loop over states
        order = np.zeros((n_states, n_turbines), dtype=FC.ITYPE)
        for si in range(n_states):
            order[si] = np.lexsort(keys=coosx[si])

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
        theights = fdata[FV.H][:, downwind_index]
        
        if len(np.unique(rxyh, axis=0)) != 1:
            raise ValueError(f"{self.name}: Turbines with downwind order {downwind_index} seem to be moving, this is incompatible with this wake frame")

        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][:, downwind_index, None]

        wcoos = np.full((n_states, n_targets, n_tpoints, 3), 1e20, dtype=FC.DTYPE)
        wcoos[..., 2] = targets[..., 2] - rxyh[:, None, None, 2]
        
        key_tracep = self.var(f"tracep_{downwind_index}")
        try:
            trace_p = algo.get_from_chunk_store(key_tracep, mdata, tdata=tdata)
        except KeyError:

            key_pv = self.var(f"pv_{downwind_index}")
            i0 = mdata.states_i0(counter=True)
            trace_p = np.zeros_like(rxyh[:, :2])
            if i0 == 0:
                p = rxyh[0, :2]
            else:
                try:
                    p, v = algo.get_from_chunk_store(key_pv, mdata, tdata=tdata, prev_s=1)
                    p += v
                except KeyError:
                    return wcoos
            trace_p[0] = p
            
            for si in range(n_states):
                htdata = TData.from_points() 
            