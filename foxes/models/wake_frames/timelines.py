import numpy as np
from xarray import Dataset

from foxes.core import WakeFrame, MData, FData, TData
from foxes.utils import wd2uv
from foxes.algorithms.iterative import Iterative
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class Timelines(WakeFrame):
    """
    Dynamic wakes for spatially uniform timeseries states.

    Attributes
    ----------
    max_length_km: float
        The maximal wake length in km
    cl_ipars: dict
        Interpolation parameters for centre line
        point interpolation
    dt_min: float
        The delta t value in minutes,
        if not from timeseries data

    :group: models.wake_frames

    """

    def __init__(self, max_length_km=2e4, cl_ipars={}, dt_min=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        max_length_km: float
            The maximal wake length in km
        cl_ipars: dict
            Interpolation parameters for centre line
            point interpolation
        dt_min: float, optional
            The delta t value in minutes,
            if not from timeseries data
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(max_length_km=max_length_km, **kwargs)
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min

    def __repr__(self):
        return f"{type(self).__name__}(dt_min={self.dt_min}, max_length_km={self.max_length_km})"

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
            dt = (
                (times[1:] - times[:-1])
                .astype("timedelta64[s]")
                .astype(config.dtype_int)
            )
        else:
            n = max(len(times) - 1, 1)
            dt = np.full(n, self.dt_min * 60, dtype="timedelta64[s]").astype(
                config.dtype_int
            )

        # prepare mdata:
        data = algo.get_model_data(states)["coords"]
        mdict = {v: np.array(d) for v, d in data.items()}
        mdims = {v: (v,) for v in data.keys()}
        data = algo.get_model_data(states)["data_vars"]
        mdict.update({v: d[1] for v, d in data.items()})
        mdims.update({v: d[0] for v, d in data.items()})
        mdata = MData(mdict, mdims, loop_dims=[FC.STATE], states_i0=0)
        del mdict, mdims, data

        # prepare fdata:
        fdata = FData({}, {}, loop_dims=[FC.STATE])

        # prepare tdata:
        n_states = states.size()
        data = {
            v: np.zeros((n_states, 1, 1), dtype=config.dtype_double)
            for v in states.output_point_vars(algo)
        }
        pdims = {v: (FC.STATE, FC.TARGET, FC.TPOINT) for v in data.keys()}
        points = np.zeros((n_states, 1, 3), dtype=config.dtype_double)

        # calculate all heights:
        self.timelines_data = {"dxy": (("height", FC.STATE, "dir"), [])}
        for h in heights:

            if verbosity > 0:
                print(f"  Height: {h} m")

            points[..., 2] = h
            tdata = TData.from_points(
                points=points,
                data=data,
                dims=pdims,
            )

            res = states.calculate(algo, mdata, fdata, tdata)
            del tdata

            uv = wd2uv(res[FV.WD], res[FV.WS])[:, 0, 0, :2]
            if len(dt) == 1:
                dxy = uv * dt[0]
            else:
                dxy = uv[:-1] * dt[:, None]
                dxy = np.append(dxy, uv[-1, None, :] * dt[-1], axis=0)
            self.timelines_data["dxy"][1].append(dxy)
            """ DEBUG
            import matplotlib.pyplot as plt
            xy = np.array([np.sum(self.timelines_data[h][:n], axis=0) for n in range(len(self.timelines_data[h]))])
            print(xy)
            plt.plot(xy[:, 0], xy[:, 1])
            plt.title(f"Height {h} m")
            plt.show()
            quit()
            """

            if needs_res:
                if "U" not in self.timelines_data:
                    self.timelines_data["U"] = (("height", FC.STATE), [])
                    self.timelines_data["V"] = (("height", FC.STATE), [])
                self.timelines_data["U"][1].append(uv[:, 0])
                self.timelines_data["V"][1].append(uv[:, 1])

                for v in states.output_point_vars(algo):
                    if v not in [FV.WS, FV.WD]:
                        if v not in self.timelines_data:
                            self.timelines_data[v] = (("height", FC.STATE), [])
                        self.timelines_data[v][1].append(res[v][:, 0, 0])

            del res, uv, dxy

        self.timelines_data = Dataset(
            coords={
                FC.STATE: states.index(),
                "height": heights,
            },
            data_vars={
                v: (d[0], np.stack(d[1], axis=0))
                for v, d in self.timelines_data.items()
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
        if not isinstance(algo, Iterative):
            raise TypeError(
                f"Incompatible algorithm type {type(algo).__name__}, expecting {Iterative.__name__}"
            )
        super().initialize(algo, verbosity)

        # find turbine hub heights:
        t2h = np.zeros(algo.n_turbines, dtype=config.dtype_double)
        for ti, t in enumerate(algo.farm.turbines):
            t2h[ti] = (
                t.H if t.H is not None else algo.farm_controller.turbine_types[ti].H
            )
        heights = np.unique(t2h)

        # pre-calc data:
        from foxes.input.states import OnePointFlowTimeseries

        if isinstance(algo.states, OnePointFlowTimeseries):
            self._precalc_data(algo, algo.states.base_states, heights, verbosity)
        else:
            self._precalc_data(algo, algo.states, heights, verbosity)

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

    def calc_order(self, algo, mdata, fdata):
        """
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
        order = np.zeros((fdata.n_states, fdata.n_turbines), dtype=config.dtype_int)
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
            in the downwind order

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
        heights = self.timelines_data["height"].to_numpy()
        data_dxy = self.timelines_data["dxy"].to_numpy()

        D = np.zeros((n_states, n_points), dtype=config.dtype_double)
        D[:] = fdata[FV.D][:, downwind_index, None]

        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=config.dtype_double)
        wcoosx = wcoos[:, :, 0]
        wcoosy = wcoos[:, :, 1]
        wcoos[:, :, 2] = points[:, :, 2] - rxyz[:, None, 2]

        i0 = mdata.states_i0(counter=True)
        i1 = i0 + mdata.n_states
        trace_si = np.zeros((n_states, n_points), dtype=config.dtype_int)
        trace_si[:] = i0 + np.arange(n_states)[:, None]
        for hi, h in enumerate(heights):
            dxy = data_dxy[hi][:i1]
            precond = theights[:, None] == h

            trace_p = np.zeros((n_states, n_points, 2), dtype=config.dtype_double)
            trace_p[:] = points[:, :, :2] - rxyz[:, None, :2]
            trace_l = np.zeros((n_states, n_points), dtype=config.dtype_double)
            trace_d = np.full((n_states, n_points), np.inf, dtype=config.dtype_double)
            h_trace_si = trace_si.copy()

            # flake8: noqa: F821
            def _update_wcoos(sel):
                """Local function that updates coordinates and source times"""
                nonlocal wcoosx, wcoosy, trace_si
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
                        wcoosx[sel] = np.where(seln, projx + trace_l[sel], wcoosx[sel])

                        ny = np.concatenate([-nx[:, 1, None], nx[:, 0, None]], axis=1)
                        projy = np.einsum("sd,sd->s", trp, ny)
                        wcoosy[sel] = np.where(seln, projy, wcoosy[sel])
                        del ny, projy

                        trace_si[sel] = np.where(seln, h_trace_si[sel], trace_si[sel])

            # step backwards in time, until wake source turbine is hit:
            _update_wcoos(precond)
            while True:
                sel = precond & (h_trace_si > 0) & (trace_l < self.max_length_km * 1e3)
                if np.any(sel):
                    h_trace_si[sel] -= 1

                    delta = dxy[h_trace_si[sel]]
                    dmag = np.linalg.norm(delta, axis=-1)
                    trace_p[sel] -= delta
                    trace_l[sel] += dmag
                    del delta, dmag

                    # check if this is closer to turbine:
                    _update_wcoos(sel)
                    del sel

                else:
                    del sel
                    break
            del trace_p, trace_l, trace_d, h_trace_si, dxy, precond

        # store turbines that cause wake:
        trace_si = np.minimum(trace_si, i0 + np.arange(n_states)[:, None])
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # store states that cause wake for each target point,
        # will be used by model.get_data() during wake calculation:
        tdata.add(
            FC.STATES_SEL,
            trace_si.reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )

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
        heights = self.timelines_data["height"].to_numpy()
        data_dxy = self.timelines_data["dxy"].to_numpy()

        points = np.zeros((n_states, n_points, 3), dtype=config.dtype_double)
        points[:] = rxyz[:, None, :]

        trace_dp = np.zeros_like(points[..., :2])
        trace_l = x.copy()
        trace_si = np.zeros((n_states, n_points), dtype=config.dtype_int)
        trace_si[:] = np.arange(n_states)[:, None]

        for hi, h in enumerate(heights):
            precond = theights == h
            if np.any(precond):
                sel = precond[:, None] & (trace_l > 0)
                while np.any(sel):
                    dxy = data_dxy[hi][trace_si[sel]]

                    trl = trace_l[sel]
                    trp = trace_dp[sel]
                    dl = np.linalg.norm(dxy, axis=-1)
                    cl = np.abs(trl - dl) < np.abs(trl)
                    if np.any(cl):
                        trace_l[sel] = np.where(cl, trl - dl, trl)
                        trace_dp[sel] = np.where(cl[:, None], trp + dxy, trp)
                    del trl, trp, dl, cl, dxy

                    trace_si[sel] -= 1
                    sel = precond[:, None] & (trace_l > 0) & (trace_si >= 0)

                si = trace_si[precond] + 1
                dxy = data_dxy[hi][si]
                dl = np.linalg.norm(dxy, axis=-1)[:, :, None]
                trl = trace_l[precond][:, :, None]
                trp = trace_dp[precond]
                sel = np.abs(trl) < 2 * dl
                trace_dp[precond] = np.where(sel, trp - trl / dl * dxy, np.nan)

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
        self.timelines_data = None
