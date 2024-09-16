import numpy as np
import pandas as pd
from xarray import Dataset
from pathlib import Path
from scipy.interpolate import interpn

from foxes.core import States, MData, FData, TData
from foxes.utils import wd2uv, uv2wd
from foxes.data import STATES
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
        
        if base_states is not None and len(base_states_kwargs):
            raise KeyError(f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_state_pars {list(base_states_kwargs.keys())}")
        elif base_states is not None and len(base_states_args):
            raise KeyError(f"Base states of type '{type(base_states).__name__}' were given, cannot handle base_states_args of types {[type(a).__name__ for a in base_states_args]}")
        elif base_states is None:
            self.base_states = States.new(*base_states_args, **base_states_kwargs)
            
        if tl_heights is None:
            if hasattr(self.base_states, "heights"):
                self.heights = self.base_states.heights
            elif len(self.ref_xy) > 2:
                self.heights = [self.ref_xy[2]]
            else:
                raise KeyError(f"Cannot find 'heights' in base states of type '{type(self.base_states).__name__}', missing either `ref_xy` of type [x, y, z], or explicit value list via parameter 'tl_heights'")

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
    
    def _precalc_data(self, algo, states, heights, verbosity, needs_res=False):
        """Helper function for pre-calculation of ambient wind vectors"""
        
        if verbosity > 0:
            print(f"{self.name}: Pre-calculating ambient wind vectors")

        # get and check times:
        times = np.asarray(states.index())
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

        # prepare mdata:
        data = algo.get_model_data(states)["coords"]
        mdict = {v: np.array(d) for v, d in data.items()}
        mdims = {v: (v,) for v in data.keys()}
        data = algo.get_model_data(states)["data_vars"]
        mdict.update({v: d[1] for v, d in data.items()})
        mdims.update({v: d[0] for v, d in data.items()})
        mdata = MData(mdict, mdims, loop_dims=[FC.STATE])
        del mdict, mdims, data

        # prepare fdata:
        fdata = FData({}, {}, loop_dims=[FC.STATE])

        # prepare tdata:
        tdata = {
            v: np.zeros((algo.n_states, 1, 1), dtype=FC.DTYPE)
            for v in states.output_point_vars(algo)
        }
        pdims = {v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in tdata.keys()}
        points = np.zeros((algo.n_states, 1, 3), dtype=FC.DTYPE)

        # calculate all heights:
        self._data = {"dxy": (("height", FC.STATE, "dir"), [])}
        for h in heights:

            if verbosity > 0:
                print(f"  Height: {h} m")

            points[..., 2] = h
            tdata = TData.from_points(
                points=points,
                data=tdata,
                dims=pdims,
            )

            res = states.calculate(algo, mdata, fdata, tdata)
            del tdata
            
            uv = wd2uv(res[FV.WD], res[FV.WS])[:, 0, 0, :2]
            if len(dt) == 1:
                dxy = uv * dt[:, None]
            else:
                dxy = uv[:-1] * dt[:, None]
                dxy = np.insert(dxy, 0, dxy[0], axis=0)
            self._data["dxy"][1].append(dxy)
            """ DEBUG
            import matplotlib.pyplot as plt
            xy = np.array([np.sum(self._data[h][:n], axis=0) for n in range(len(self._data[h]))])
            print(xy)
            plt.plot(xy[:, 0], xy[:, 1])
            plt.title(f"Height {h} m")
            plt.show()
            quit()
            """
            
            if needs_res:
                if "U" not in self._data:
                    self._data[v] = {"U": (("height", FC.STATE), [])}
                    self._data[v] = {"V": (("height", FC.STATE), [])}
                self._data["U"][1].append(uv[:, 0])
                self._data["V"][1].append(uv[:, 1])
                
                for v in states.output_point_vars(algo):
                    if v not in [FV.WS, FV.WD]:
                        if v not in self._data:
                            self._data[v] = {v: (("height", FC.STATE), [])}
                        self._data[v][1].append(res[v][:, 0, 0])
                        
            del res, uv, dxy
                        
        self._data = Dataset(
            coords={
                FC.STATE: states.index(),
                "height": heights,
            },
            data_vars={
                v: (d[0], np.stack(d[1], axis=0))
                for v, d in self._data.items()
            },
        )

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

        # pre-calc data:
        self._precalc_data(algo, self.base_states, self.heights, verbosity, needs_res=True)
            
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
        return self.base_states.output_point_vars()

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
        return self.base_states.weights()

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
            data_stash[self.name]["data"] = self._data

            if isel is not None:
                self._data = self._data.isel(isel)
            if sel is not None:
                self._data = self._data.sel(sel)

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
            self._data = data.pop("data")
            
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
        heights = self._data["height"].to_numpy()
        data_dxy = self._data["dxy"].to_numpy()
        n_heights = len(heights)

        # compute states indices for all requested points:
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + n_states
        trace_si = np.zeros((n_heights, n_states, n_points), dtype=FC.ITYPE)
        trace_si[:] = i0 + np.arange(n_states)[None, :, None] + 1
        trace_done = np.zeros((n_heights, n_states, n_points), dtype=bool)
        for hi in range(n_heights):
            
            trace_p = points[:, :, :2] - self.ref_xy[None, None, :2]
            h_trace_si = trace_si[hi]

            # step backwards in time, until projection onto axis is negative:
            while np.any(~trace_done[hi]):
                sel = ~trace_done[hi]
                h_trace_si[sel] -= 1
                
                nx = data_dxy[hi][trace_si[sel]]
                trace_p[sel] -= nx
                nx /= np.linalg.norm(nx, axis=-1)
                projx = np.einsum("sd,sd->s", trace_p[sel], nx)
                
                seld = (projx < 0) | (h_trace_si[sel] < 0)
                if np.any(seld):
                    trd = trace_done[hi][sel]
                    trd[seld] = True
                    trace_done[hi][sel] = trd
                    
                    del trd
                del seld, projx, nx, sel
            del h_trace_si, trace_p
        del trace_done
                
        # interpolate to heights:
        if n_heights > 1:
            
            ar_states = np.arange(n_states)
            ar_points = np.arange(n_points)
            
            data = {v: d.to_numpy()[trace_si]
                    for v, d in self._data.data_vars.items()
                    if v != "dxy"}
            vres = list(data.keys())
            data = np.stack(data.values(), axis=-1)
            
            eval = np.zeros((n_states, n_points, 3), dtype=FC.DTYPE)
            eval[:, :, 0] = points[:, :, 2]
            eval[:, :, 1] = ar_states[:, None]
            eval[:, :, 2] = ar_points[None, :]
            
            ires = interpn(
                (heights, ar_states, ar_points), data, eval)
            del eval, data, ar_points, ar_states
            
            results = {}
            for v in self.output_point_vars(algo):
                if v in [FV.WS, FV.WD] and v not in results:
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
                    
                else:
                    results[v] = ires[:, :, vres.index(v)]
        
        # no dependence on height:
        else:
            results = {}
            sel = trace_si[0]
            for v in self.output_point_vars(algo):
                if v in [FV.WS, FV.WD] and v not in results:
                    uv = np.stack(
                        [self._data["U"].to_numpy()[0, sel], 
                        self._data["V"].to_numpy()[0, sel]],
                        axis=-1
                    )
                    results = {
                        FV.WD: uv2wd(uv),
                        FV.WS: np.linalg.norm(uv, axis=-1)
                    }
                    del uv
                    
                else:
                    results[v] = self._data[v].to_numpy()[0, sel]
        
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
        super().__init__(*args, states_type="MultiHeightStates", **kwargs)
        
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
        super().__init__(*args, states_type="MultiHeightNCStates", **kwargs)
        