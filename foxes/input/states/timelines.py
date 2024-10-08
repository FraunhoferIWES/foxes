import numpy as np
from scipy.interpolate import interpn

from foxes.core import States
from foxes.utils import uv2wd
from foxes.models.wake_frames.timelines import Timelines
import foxes.variables as FV
import foxes.constants as FC

class TimelinesStates(States):
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
        self.ref_xy = np.array(ref_xy, dtype=FC.DTYPE)
        self.heights = tl_heights
        self.base_states = base_states
        self.dt_min = dt_min
        
        self.intp_pars = {"fill_value": None}
        if "bounds_error" in base_states_kwargs:
            self.intp_pars["bounds_error"] = base_states_kwargs["bounds_error"]
        
        if base_states is not None and len(base_states_kwargs):
            raise KeyError(f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_state_pars {list(base_states_kwargs.keys())}")
        elif base_states is not None and len(base_states_args):
            raise KeyError(f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_states_args of types {[type(a).__name__ for a in base_states_args]}")
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
                raise KeyError(f"Cannot find 'heights' in base states of type '{type(self.base_states).__name__}', missing either `ref_xy` of type [x, y, z], or explicit value list via parameter 'tl_heights'")
        
        # pre-calc data:
        Timelines._precalc_data(
            self, algo, self.base_states, self.heights, verbosity, needs_res=True)
            
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
            
    def calc_states_indices(self, mdata, points, hi, ref_xy):
        
        n_states, n_points = points.shape[:2]
        dxy = self.timelines_data["dxy"].to_numpy()[hi]
        
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + n_states
        trace_p = points[:, :, :2] - ref_xy[:, :, :2]
        trace_si = np.zeros((n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = i0 + np.arange(n_states)[:, None] 
        trace_done = np.zeros((n_states, n_points), dtype=bool)
        
        # flake8: noqa: F821
        def _eval_trace(sel, projx0=None, hdxy=None, trs=None):
            """Helper function that updates trace_done"""
            nonlocal trace_si, trace_done
            
            trs = trace_si[sel] if trs is None else trs
            nx = dxy[trs] if hdxy is None else hdxy
            lx = np.linalg.norm(nx, axis=-1)
            nx /= lx[..., None]
            projx = np.einsum("...d,...d->...", trace_p[sel], nx)
 
            # find crossings of planes orthogonal to hdxy:
            if projx0 is not None:
                print("EVAL",np.sum(projx0 > -1e-8),np.sum(projx < 1e-8),np.sum(projx0 < 1e-8),np.sum(projx > -1e-8),":",np.sum((projx0 > -1e-8) & (projx < 1e-8)),np.sum((projx0 < 1e-8) & (projx > -1e-8)))
                seld = (
                    (projx0 > -1e-8) & (projx < 1e-8)
                ) | (
                    (projx0 < 1e-8) & (projx > -1e-8)
                )
                if np.any(seld):
                    trace_done[sel] = np.where(seld, True, trace_done[sel])
            
            return projx

        # step backwards in time, until projection onto axis is negative:
        projx0 = _eval_trace(sel=np.s_[:])
        """
        sel = (~trace_done)
        while np.any(sel):
            
            trs = trace_si[sel]
            hdxy = dxy[trs]
            trace_p[sel] -= hdxy
            
            projx0[sel] = _eval_trace(
                sel, projx0=projx0[sel], hdxy=hdxy, trs=trs)
            
            trace_si[~trace_done & sel] -= 1
            sel = (~trace_done) & (trace_si>=0)
            
            del trs, hdxy
        """
        
        # step forwards in time, until projection onto axis is positive:
        sel = (~trace_done)
        print("SEL",np.sum(sel))
        if np.any(sel):
            trace_p = np.where(sel[:, :, None], points[:, :, :2] - ref_xy[:, :, :2], trace_p)
            trace_si = np.where(sel, i0 + np.arange(n_states)[:, None] + 1, trace_si)
                
            sel &= (trace_si<i1)
            projx0[sel] = _eval_trace(sel)
            while np.any(sel):
                
                trs = trace_si[sel]
                hdxy = dxy[trs]
                trace_p[sel] += hdxy
                
                projx0[sel] = _eval_trace(
                    sel, projx0=projx0[sel], hdxy=hdxy, trs=trs)
                
                trace_si[~trace_done & sel] += 1
                sel = (~trace_done) & (trace_si<i1)
                
                del trs, hdxy     
        
        print("DONE",np.sum(trace_done), np.sum(~trace_done),np.sum(trace_si<0),np.sum(trace_si>=i1))   
            
        trace_si = np.maximum(trace_si, 0)
        trace_si = np.minimum(trace_si, i1 - 1)
    
        return trace_si
            
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
        for hi in range(n_heights):
            trace_si.append(self.calc_states_indices(
                mdata, points, hi, self.ref_xy[None, None, :]
            ))
                
        # interpolate to heights:
        if n_heights > 1:
            
            ar_states = np.arange(n_states)
            ar_points = np.arange(n_points)
            
            crds = (heights, ar_states, ar_points)
            
            data = {v: np.stack(
                        [d.to_numpy()[hi, trace_si[hi]] for hi in range(n_heights)]
                        , axis=0)
                    for v, d in self.timelines_data.data_vars.items()
                    if v != "dxy"}
            vres = list(data.keys())
            data = np.stack(list(data.values()), axis=-1)
            
            eval = np.zeros((n_states, n_points, 3), dtype=FC.DTYPE)
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
                        [ires[:, :, vres.index("U")], 
                        ires[:, :, vres.index("V")]],
                        axis=-1
                    )
                    results = {
                        FV.WD: uv2wd(uv),
                        FV.WS: np.linalg.norm(uv, axis=-1)
                    }
                    del uv
        
        # no dependence on height:
        else:
            results = {}
            sel = trace_si[0]
            for v in self.output_point_vars(algo):
                if v not in [FV.WS, FV.WD]:
                    results[v] = self.timelines_data[v].to_numpy()[0, sel]
                elif v not in results:
                    uv = np.stack(
                        [self.timelines_data["U"].to_numpy()[0, sel], 
                        self.timelines_data["V"].to_numpy()[0, sel]],
                        axis=-1
                    )
                    results = {
                        FV.WD: uv2wd(uv),
                        FV.WS: np.linalg.norm(uv, axis=-1)
                    }
                    del uv
        
        return {v: d.reshape(n_states, n_targets, n_tpoints)
                for v, d in results.items()}

class TimeseriesTimelines(TimelinesStates):
    """
    Timelines from uniform timeseries data
    
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
            tl_heights = [100.]
        super().__init__(
            ref_xy, 
            *args, 
            tl_heights=tl_heights, 
            states_type="Timeseries", 
            **kwargs,
        )

class MultiHeightTimelines(TimelinesStates):
    """
    Timelines from height dependent timeseries data
    
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
        
class MultiHeightNCTimelines(TimelinesStates):
    """
    Timelines from height dependent timeseries data
    
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
        