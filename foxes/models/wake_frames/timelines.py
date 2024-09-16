import numpy as np
from xarray import Dataset

from foxes.core import WakeFrame,MData, FData, TData
from foxes.utils import wd2uv
from foxes.input.states import TimelinesStates
import foxes.variables as FV
import foxes.constants as FC


class Timelines(WakeFrame):
    """
    Dynamic wakes for spatially uniform timeseries states.

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
        heights = np.unique(t2h)

        # pre-calc data:
        TimelinesStates._precalc_data(self, algo, algo.states, heights, verbosity)

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        TimelinesStates.set_running(self, algo, data_stash, sel, isel, verbosity)

    def unset_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        TimelinesStates.unset_running(self, algo, data_stash, sel, isel, verbosity)

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
        points = targets.reshape(n_states, n_points, 3)
        rxyz = fdata[FV.TXYH][:, downwind_index]
        theights = fdata[FV.H][:, downwind_index]
        heights = self._data["height"].to_numpy()
        data_dxy = self._data["dxy"].to_numpy()

        D = np.zeros((n_states, n_points), dtype=FC.DTYPE)
        D[:] = fdata[FV.D][:, downwind_index, None]

        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=FC.DTYPE)
        wcoosx = wcoos[:, :, 0]
        wcoosy = wcoos[:, :, 1]
        wcoos[:, :, 2] = points[:, :, 2] - rxyz[:, None, 2]

        i0 = mdata.states_i0(counter=True)
        i1 = i0 + mdata.n_states
        trace_si = np.zeros((n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = i0 + np.arange(n_states)[:, None] + 1
        for hi, h in enumerate(heights):

            trace_p = np.zeros((n_states, n_points, 2), dtype=FC.DTYPE)
            trace_p[:] = points[:, :, :2] - rxyz[:, None, :2]
            trace_l = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            trace_d = np.full((n_states, n_points), np.inf, dtype=FC.DTYPE)
            h_trace_si = trace_si.copy()

            # step backwards in time, until wake source turbine is hit:
            dxy = data_dxy[hi][:i1]
            precond = theights[:, None] == h
            while True:
                sel = precond & (h_trace_si > 0) & (trace_l < self.max_wake_length)
                if np.any(sel):
                    h_trace_si[sel] -= 1

                    delta = dxy[h_trace_si[sel]]
                    dmag = np.linalg.norm(delta, axis=-1)
                    trace_p[sel] -= delta
                    trace_l[sel] += dmag
                    del delta, dmag

                    # check if this is closer to turbine:
                    d = np.linalg.norm(trace_p, axis=-1)
                    sel = sel & (d <= trace_d)
                    if np.any(sel):
                        trace_d[sel] = d[sel]

                        nx = dxy[h_trace_si[sel]]
                        dx = np.linalg.norm(nx, axis=-1)
                        nx /= dx[:, None]
                        trp = trace_p[sel]
                        projx = np.einsum("sd,sd->s", trp, nx)

                        seln = (projx > -dx) & (projx < dx)
                        if np.any(seln):
                            wcoosx[sel] = np.where(
                                seln, projx + trace_l[sel], wcoosx[sel]
                            )

                            ny = np.concatenate(
                                [-nx[:, 1, None], nx[:, 0, None]], axis=1
                            )
                            projy = np.einsum("sd,sd->s", trp, ny)
                            wcoosy[sel] = np.where(seln, projy, wcoosy[sel])
                            del ny, projy

                            trace_si[sel] = np.where(
                                seln, h_trace_si[sel], trace_si[sel]
                            )

                        del nx, dx, projx, seln
                    del d, sel

                else:
                    break

            del trace_p, trace_l, trace_d, h_trace_si, dxy, precond

        # store turbines that cause wake:
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # store states that cause wake for each target point,
        # will be used by model.get_data() during wake calculation:
        tdata.add(
            FC.STATES_SEL,
            trace_si.reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )
        
        wcoos[wcoos>9e19] = np.nan

        return wcoos.reshape(n_states, n_targets, n_tpoints, 3)

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        # prepare:
        n_states, n_points = x.shape
        rxyz = fdata[FV.TXYH][:, downwind_index]
        theights = fdata[FV.H][:, downwind_index]
        heights = self._data["height"].to_numpy()
        data_dxy = self._data["dxy"].to_numpy()
        
        points = np.zeros((n_states, n_points, 3), dtype=FC.DTYPE)
        points[:] = rxyz[:, None, :]
        
        trace_dp = np.zeros_like(points[..., :2])
        trace_l = x.copy()
        trace_si = np.zeros((n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = np.arange(n_states)[:, None]
          
        for hi, h in enumerate(heights):
            precond = (theights==h)
            if np.any(precond):
                sel = precond[:, None] & (trace_l>0)
                while np.any(sel):
                    dxy = data_dxy[hi][trace_si[sel]]
            
                    trl = trace_l[sel]
                    trp = trace_dp[sel]
                    dl = np.linalg.norm(dxy, axis=-1)
                    cl = np.abs(trl-dl) < np.abs(trl)
                    if np.any(cl):
                        trace_l[sel] = np.where(cl, trl-dl, trl)
                        trace_dp[sel] = np.where(cl[:, None], trp+dxy, trp)
                    del trl, trp, dl, cl, dxy
                    
                    trace_si[sel] -= 1
                    sel = precond[:, None] & (trace_l>0) & (trace_si>=0)
                
                si = trace_si[precond] + 1
                dxy = data_dxy[hi][si]
                dl = np.linalg.norm(dxy, axis=-1)[:, :, None]
                trl = trace_l[precond][:, :, None]
                trp = trace_dp[precond]
                sel = np.abs(trl) < 2*dl
                trace_dp[precond] = np.where(sel, trp-trl/dl*dxy, np.nan)
                
                del si, dxy, dl, trl, trp, sel
            del precond
        del trace_si, trace_l
        
        points[..., :2] += trace_dp
        
        return points
                    
    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().finalize(algo, verbosity=verbosity)
        self._data = None
