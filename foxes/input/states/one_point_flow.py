import numpy as np
from scipy.interpolate import interpn

from foxes.core import States
from foxes.utils import uv2wd
from foxes.models.wake_frames.timelines import Timelines
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class OnePointFlowStates(States):
    """
    Time-evolving states based on horizontally
    homogeneous timeseries data

    Attributes
    ----------
    ref_xy: list of float
        The [x, y] or [x, y, z] coordinates of the base states.
        If [x, y, z] then z will serve as height
    tl_heights: list of float
        The heights at which timelines will be calculated
    dt_min: float
        The delta t value in minutes,
        if not from timeseries data
    intp_pars: dict
        Parameters for height interpolation with
        scipy.interpolate.interpn

    :group: input.states

    """

    def __init__(
        self,
        ref_xy,
        *base_states_args,
        base_states=None,
        tl_heights=None,
        dt_min=None,
        **base_states_kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        ref_xy: list of float
            The [x, y] or [x, y, z] coordinates of the base states.
            If [x, y, z] then z will serve as height
        base_states_args: tuple, optional
            Arguments for creating the base states from
            States.new(), if not given as base_states
        base_states: foxes.core.States, optional
            The base states, representing horizontally
            homogeneous inflow
        tl_heights: list of float, optional
            The heights at which timelines will be calculated
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data
        base_states_kwargs: dict, optional
            Arguments for creating the base states from
            States.new(), if not given as base_states

        """
        super().__init__()
        self.ref_xy = np.array(ref_xy, dtype=config.dtype_double)
        self.heights = tl_heights
        self.base_states = base_states
        self.dt_min = dt_min

        self.intp_pars = {"fill_value": None}
        if "bounds_error" in base_states_kwargs:
            self.intp_pars["bounds_error"] = base_states_kwargs["bounds_error"]

        if base_states is not None and len(base_states_kwargs):
            raise KeyError(
                f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_state_pars {list(base_states_kwargs.keys())}"
            )
        elif base_states is not None and len(base_states_args):
            raise KeyError(
                f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_states_args of types {[type(a).__name__ for a in base_states_args]}"
            )
        elif base_states is None:
            self.base_states = States.new(*base_states_args, **base_states_kwargs)

    def __repr__(self):
        return f"{type(self).__name__}(base={type(self.base_states).__name__}, heights={self.heights}, dt_min={self.dt_min})"

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.base_states]

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

        # find heights:
        if self.heights is None:
            if hasattr(self.base_states, "heights"):
                self.heights = self.base_states.heights
            elif len(self.ref_xy) > 2:
                self.heights = [self.ref_xy[2]]
            else:
                raise KeyError(
                    f"Cannot find 'heights' in base states of type '{type(self.base_states).__name__}', missing either `ref_xy` of type [x, y, z], or explicit value list via parameter 'tl_heights'"
                )

        # pre-calc data:
        Timelines._precalc_data(
            self, algo, self.base_states, self.heights, verbosity, needs_res=True
        )

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.base_states.size()

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return self.base_states.index()

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return self.base_states.output_point_vars(algo)

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights: numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        return self.base_states.weights(algo)

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
        super().set_running(algo, data_stash, sel, isel, verbosity)

        if sel is not None or isel is not None:
            data_stash[self.name]["data"] = self.timelines_data

            if isel is not None:
                self.timelines_data = self.timelines_data.isel(isel)
            if sel is not None:
                self.timelines_data = self.timelines_data.sel(sel)

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
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        data = data_stash[self.name]
        if "data" in data:
            self.timelines_data = data.pop("data")

    def calc_states_indices(self, algo, mdata, points, hi, ref_xy):

        n_states, n_points = points.shape[:2]
        dxy = self.timelines_data["dxy"].to_numpy()[hi]

        i0 = mdata.states_i0(counter=True)
        trace_p = points[:, :, :2] - ref_xy[:, :, :2]
        trace_si = np.zeros((n_states, n_points), dtype=config.dtype_int)
        trace_si[:] = i0 + np.arange(n_states)[:, None]
        coeffs = np.full((n_states, n_points), np.nan, dtype=config.dtype_double)

        # flake8: noqa: F821
        def _eval_trace(sel, hdxy=None, hdxy0=None, trs=None):
            """Helper function that updates trace_done"""
            nonlocal coeffs

            # project onto local x direction:
            hdxy0 = dxy[trace_si[sel]] if hdxy0 is None else hdxy0
            nx = hdxy0 / np.linalg.norm(hdxy0, axis=-1)[..., None]
            projx = np.einsum("...d,...d->...", trace_p[sel], nx)

            # check for local points:
            if hdxy is None:
                seld = (projx >= 1e-10) & (projx <= 1e-10)
                if np.any(seld):
                    coeffs[sel] = np.where(seld, -1, coeffs[sel])

            # check for vicinity to reference plane:
            else:
                lx = np.einsum("...d,...d->...", hdxy, nx)
                seld = ((lx < 0) & (projx >= lx) & (projx <= 0)) | (
                    (lx > 0) & (projx >= 0) & (projx <= lx)
                )
                if np.any(seld):
                    w = projx / np.abs(lx)
                    coeffs[sel] = np.where(seld, w, coeffs[sel])

        # step backwards in time, until projection onto axis is negative:
        _eval_trace(sel=np.s_[:])
        sel = np.isnan(coeffs)
        tshift = 0
        while np.any(sel):

            trs = trace_si[sel]
            hdxy = -dxy[trs]
            trace_p[sel] += hdxy

            _eval_trace(sel, hdxy=hdxy, hdxy0=dxy[trs - tshift])

            tshift -= 1
            sel0 = np.isnan(coeffs)
            trace_si[sel0 & sel] -= 1
            sel = sel0 & (trace_si >= 0)

            del trs, sel0, hdxy

        # step forwards in time, until projection onto axis is positive:
        sel = np.isnan(coeffs)
        if np.any(sel):
            trace_p = np.where(
                sel[:, :, None], points[:, :, :2] - ref_xy[:, :, :2], trace_p
            )
            trace_si = np.where(sel, i0 + np.arange(n_states)[:, None] + 1, trace_si)

            sel &= trace_si < algo.n_states
            tshift = 1
            while np.any(sel):

                trs = trace_si[sel]
                hdxy = dxy[trs]
                trace_p[sel] += hdxy

                _eval_trace(sel, hdxy=hdxy, hdxy0=dxy[trs - tshift])

                tshift += 1
                sel0 = np.isnan(coeffs)
                trace_si[sel0 & sel] += 1
                sel = sel0 & (trace_si < algo.n_states)

                del trs, sel0, hdxy

        return trace_si, coeffs

    def calculate(self, algo, mdata, fdata, tdata):
        """
        The main model calculation.

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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        # prepare:
        targets = tdata[FC.TARGETS]
        n_states, n_targets, n_tpoints = targets.shape[:3]
        n_points = n_targets * n_tpoints
        points = targets.reshape(n_states, n_points, 3)
        heights = self.timelines_data["height"].to_numpy()
        n_heights = len(heights)

        # compute states indices for all requested points:
        trace_si = []
        coeffs = []
        for hi in range(n_heights):
            s, c = self.calc_states_indices(
                algo, mdata, points, hi, self.ref_xy[None, None, :]
            )
            trace_si.append(s)
            coeffs.append(c)
            del s, c

        # flake8: noqa: F821
        def _interp_time(hi, v):
            """Helper function for interpolation bewteen states"""

            sts = trace_si[hi]
            cfs = coeffs[hi]
            data = self.timelines_data[v].to_numpy()[hi]
            out = np.zeros(sts.shape, dtype=config.dtype_double)

            sel_low = sts < 0
            if np.any(sel_low):
                out[sel_low] = data[0]

            sel_hi = sts >= algo.n_states
            if np.any(sel_hi):
                out[sel_hi] = data[algo.n_states - 1]

            sel = (~sel_low) & (~sel_hi) & (cfs <= 0)
            if np.any(sel):
                s = sts[sel]
                c = -cfs[sel]
                out[sel] = c * data[s] + (1 - c) * data[s - 1]

            sel = (~sel_low) & (~sel_hi) & (cfs > 0)
            if np.any(sel):
                s = sts[sel]
                c = cfs[sel]
                out[sel] = c * data[s - 1] + (1 - c) * data[s]

            return out

        # interpolate to heights:
        if n_heights > 1:

            ar_states = np.arange(n_states)
            ar_points = np.arange(n_points)

            crds = (heights, ar_states, ar_points)

            data = {
                v: np.stack([_interp_time(hi, v) for hi in range(n_heights)], axis=0)
                for v in self.timelines_data.data_vars.keys()
                if v != "dxy"
            }
            vres = list(data.keys())
            data = np.stack(list(data.values()), axis=-1)

            eval = np.zeros((n_states, n_points, 3), dtype=config.dtype_double)
            eval[:, :, 0] = points[:, :, 2]
            eval[:, :, 1] = ar_states[:, None]
            eval[:, :, 2] = ar_points[None, :]

            try:
                ires = interpn(crds, data, eval, **self.intp_pars)
            except ValueError as e:
                print(f"\nStates '{self.name}': Interpolation error")
                print("INPUT VARS: (heights, states, points)")
                print(
                    "DATA BOUNDS:",
                    [float(np.min(d)) for d in crds],
                    [float(np.max(d)) for d in crds],
                )
                print(
                    "EVAL BOUNDS:",
                    [float(np.min(p)) for p in eval.T],
                    [float(np.max(p)) for p in eval.T],
                )
                print(
                    "\nMaybe you want to try the option 'bounds_error=False'? This will extrapolate the data.\n"
                )
                raise e
            del crds, eval, data, ar_points, ar_states

            results = {}
            for v in self.output_point_vars(algo):
                if v not in [FV.WS, FV.WD]:
                    results[v] = ires[:, :, vres.index(v)]
                elif v not in results:
                    uv = np.stack(
                        [ires[:, :, vres.index("U")], ires[:, :, vres.index("V")]],
                        axis=-1,
                    )
                    results = {FV.WD: uv2wd(uv), FV.WS: np.linalg.norm(uv, axis=-1)}
                    del uv

        # no dependence on height:
        else:
            results = {}
            for v in self.output_point_vars(algo):
                if v not in [FV.WS, FV.WD]:
                    results[v] = _interp_time(hi, v)
                elif v not in results:
                    uv = np.stack(
                        [_interp_time(hi, "U"), _interp_time(hi, "V")], axis=-1
                    )
                    results = {FV.WD: uv2wd(uv), FV.WS: np.linalg.norm(uv, axis=-1)}
                    del uv

        return {
            v: d.reshape(n_states, n_targets, n_tpoints) for v, d in results.items()
        }


class OnePointFlowTimeseries(OnePointFlowStates):
    """
    Inhomogeneous inflow from homogeneous timeseries data
    at one point

    :group: input.states

    """

    def __init__(self, ref_xy, *args, tl_heights=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        ref_xy: list of float
            The [x, y] or [x, y, z] coordinates of the base states.
            If [x, y, z] then z will serve as height
        args: tuple, optional
            Parameters for the base class
        tl_heights: list of float, optional
            The heights at which timelines will be calculated
        kwargs: dict, optional
            Parameters for the base class

        """
        if tl_heights is None and len(ref_xy) < 3:
            tl_heights = [100.0]
        super().__init__(
            ref_xy,
            *args,
            tl_heights=tl_heights,
            states_type="Timeseries",
            **kwargs,
        )


class OnePointFlowMultiHeightTimeseries(OnePointFlowStates):
    """
    Inhomogeneous inflow from height dependent homogeneous
    timeseries data at one point

    :group: input.states

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the base class
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(*args, states_type="MultiHeightTimeseries", **kwargs)


class OnePointFlowMultiHeightNCTimeseries(OnePointFlowStates):
    """
    Inhomogeneous inflow from height dependent homogeneous
    timeseries data at one point based on NetCDF input

    :group: input.states

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the base class
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(*args, states_type="MultiHeightNCTimeseries", **kwargs)
