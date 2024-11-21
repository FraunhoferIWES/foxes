import numpy as np
from scipy.spatial.distance import cdist

from foxes.core import WakeFrame, TData
from foxes.utils import wd2uv
from foxes.algorithms.iterative import Iterative
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class DynamicWakes(WakeFrame):
    """
    Dynamic wakes for any kind of timeseries states.

    Attributes
    ----------
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

    def __init__(
        self,
        max_length_km=20,
        max_age=None,
        max_age_mean_ws=5,
        cl_ipars={},
        dt_min=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        max_length_km: float
            The maximal wake length in km
        max_age: int, optional
            The maximal number of wake steps
        max_age_mean_ws: float
            The mean wind speed for the max_age calculation,
            if the latter is not given
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

        self.max_age = max_age
        self.cl_ipars = cl_ipars
        self.dt_min = dt_min
        self._mage_ws = max_age_mean_ws

    def __repr__(self):
        return f"{type(self).__name__}(dt_min={self.dt_min}, max_length_km={self.max_length_km}, max_age={self.max_age})"

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
            self._dt = (
                (times[1:] - times[:-1])
                .astype("timedelta64[s]")
                .astype(config.dtype_int)
            )
        else:
            n = max(len(times) - 1, 1)
            self._dt = np.full(n, self.dt_min * 60, dtype="timedelta64[s]").astype(
                config.dtype_int
            )
        self._dt = np.append(self._dt, self._dt[-1, None], axis=0)

        # find max age if not given:
        if self.max_age is None:
            step = np.mean(self._mage_ws * self._dt)
            self.max_age = max(int(self.max_length_km * 1e3 / step), 1)
            if verbosity > 0:
                print(
                    f"{self.name}: Assumed mean step = {step} m, setting max_age = {self.max_age}"
                )

        self.DATA = self.var("data")
        self.UPDATE = self.var("update")

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

    def _calc_wakes(self, algo, mdata, fdata, downwind_index):
        """Helper function that computes the dynamic wakes"""
        # prepare:
        n_states = mdata.n_states
        rxyh = fdata[FV.TXYH][:, downwind_index]
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + n_states
        dt = self._dt[i0:i1]
        tdi = {
            v: (FC.STATE, FC.TARGET, FC.TPOINT)
            for v in algo.states.output_point_vars(algo)
        }
        key = f"{self.DATA}_{downwind_index}"
        ukey_fun = lambda fr, to: f"{self.UPDATE}_dw{downwind_index}_from_{fr}_to_{to}"

        # compute wakes that start within this chunk: x, y, z, length
        data = algo.get_from_chunk_store(name=key, mdata=mdata, error=False)
        if data is None:
            data = np.full(
                (n_states, self.max_age, 4), np.nan, dtype=config.dtype_double
            )
            data[:, 0, :3] = rxyh
            data[:, 0, 3] = 0
            tdt = {
                v: np.zeros((n_states, 1, 1), dtype=config.dtype_double)
                for v in tdi.keys()
            }
            pts = data[:, 0, :3].copy()
            for age in range(self.max_age - 1):
                if age == n_states:
                    break
                elif age == 0:
                    hmdata = mdata
                    hfdata = fdata
                    htdata = TData.from_points(points=pts[:, None], data=tdt, dims=tdi)
                    hdt = dt[:, None]
                else:
                    s = np.s_[age:]
                    pts = pts[:-1]
                    hmdata = mdata.get_slice(FC.STATE, s)
                    hfdata = fdata.get_slice(FC.STATE, s)
                    htdt = {v: d[s] for v, d in tdt.items()}
                    htdata = TData.from_points(points=pts[:, None], data=htdt, dims=tdi)
                    hdt = dt[s, None]
                    del htdt, s

                res = algo.states.calculate(algo, hmdata, hfdata, htdata)
                del hmdata, hfdata, htdata

                uv = wd2uv(res[FV.WD], res[FV.WS])[:, 0, 0]
                dxy = uv * hdt
                pts[:, :2] += dxy
                s = np.s_[:-age] if age > 0 else np.s_[:]
                data[s, age + 1, :3] = pts
                data[s, age + 1, 3] = data[s, age, 3] + np.linalg.norm(dxy, axis=-1)

                if age < self.max_age - 2:
                    s = ~np.isnan(data[:, age + 1, 3])
                    if np.min(data[s, age + 1, 3]) >= self.max_length_km * 1e3:
                        break

                del res, uv, s, hdt, dxy
            del pts, tdt

            # store this chunk's results:
            algo.add_to_chunk_store(key, data, mdata, copy=False)
            algo.block_convergence(mdata=mdata)

        # apply updates from future chunks:
        for (j, t), cdict in algo.chunk_store.items():
            uname = ukey_fun(j, i0)
            if j > i0 and t == 0 and uname in cdict:
                u = cdict[uname]
                if u is not None:
                    sel = np.isnan(data) & ~np.isnan(u)
                    if np.any(sel):
                        data[:] = np.where(sel, u, data)
                        algo.block_convergence(mdata=mdata)
                    cdict[uname] = None
                    del sel
                del u

        # compute wakes from previous chunks:
        prev = 0
        wi0 = i0
        data = [data]
        while True:
            prev += 1

            # read data from previous chunk:
            hdata, (h_i0, h_n_states, __, __) = algo.get_from_chunk_store(
                name=key, mdata=mdata, prev_s=prev, ret_inds=True, error=False
            )
            if hdata is None:
                break
            else:
                hdata = hdata.copy()
                wi0 = h_i0

                # select points with index+age=i0:
                sts = np.arange(h_n_states)
                ags = i0 - (h_i0 + sts)
                sel = ags < self.max_age - 1
                if np.any(sel):
                    sts = sts[sel]
                    ags = ags[sel]
                    pts = hdata[sts, ags, :3]
                    sel = (
                        np.all(~np.isnan(pts[:, :2]), axis=-1)
                        & np.any(np.isnan(hdata[sts, ags + 1, :2]), axis=-1)
                        & (hdata[sts, ags, 3] <= self.max_length_km * 1e3)
                    )
                    if np.any(sel):
                        sts = sts[sel]
                        ags = ags[sel]
                        pts = pts[sel]
                        n_pts = len(pts)

                        tdt = {
                            v: np.zeros((n_states, n_pts, 1), dtype=config.dtype_double)
                            for v in algo.states.output_point_vars(algo)
                        }

                        # compute single state wake propagation:
                        isnan0 = np.isnan(hdata)
                        for si in range(n_states):

                            s = slice(si, si + 1, None)
                            hmdata = mdata.get_slice(FC.STATE, s)
                            hfdata = fdata.get_slice(FC.STATE, s)
                            htdt = {v: d[s] for v, d in tdt.items()}
                            htdata = TData.from_points(
                                points=pts[None, :], data=htdt, dims=tdi
                            )
                            hdt = dt[s, None]
                            del htdt, s

                            res = algo.states.calculate(algo, hmdata, hfdata, htdata)
                            del hmdata, hfdata, htdata

                            uv = wd2uv(res[FV.WD], res[FV.WS])[0, :, 0]
                            dxy = uv * hdt
                            pts[:, :2] += dxy
                            del res, uv, hdt

                            ags += 1
                            hdata[sts, ags, :3] = pts
                            hdata[sts, ags, 3] = hdata[
                                sts, ags - 1, 3
                            ] + np.linalg.norm(dxy, axis=-1)
                            del dxy

                            hsel = (h_i0 + sts + ags < i1) & (ags < self.max_age - 1)
                            if np.any(hsel):
                                sts = sts[hsel]
                                ags = ags[hsel]
                                pts = pts[hsel]
                                tdt = {v: d[:, hsel] for v, d in tdt.items()}
                                del hsel
                            else:
                                del hsel
                                break

                        # store update:
                        sel = isnan0 & (~np.isnan(hdata))
                        if np.any(sel):
                            udata = np.full_like(hdata, np.nan)
                            udata[sel] = hdata[sel]
                            algo.add_to_chunk_store(
                                ukey_fun(i0, h_i0), udata, mdata=mdata, copy=False
                            )
                            algo.block_convergence(mdata=mdata)

                        del udata, tdt
                    del pts

                # store prev chunk's results:
                data.insert(0, hdata)

                del sts, ags, sel
            del hdata

        return np.concatenate(data, axis=0), wi0

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
        # first compute dynamic wakes:
        wdata, wi0 = self._calc_wakes(algo, mdata, fdata, downwind_index)

        # prepare:
        targets = tdata[FC.TARGETS]
        n_states, n_targets, n_tpoints = targets.shape[:3]
        n_points = n_targets * n_tpoints
        points = targets.reshape(n_states, n_points, 3)
        rxyh = fdata[FV.TXYH][:, downwind_index]
        i0 = mdata.states_i0(counter=True)

        # initialize:
        wcoos = np.full((n_states, n_points, 3), 1e20, dtype=config.dtype_double)
        wcoos[:, :, 2] = points[:, :, 2] - rxyh[:, None, 2]
        wake_si = np.zeros((n_states, n_points), dtype=config.dtype_int)
        wake_si[:] = i0 + np.arange(n_states)[:, None]

        # find nearest wake point:
        for si in range(n_states):
            ags = np.arange(self.max_age)
            sts = i0 + si - ags - wi0
            sel = (sts >= 0) & (sts < len(wdata))
            if np.any(sel):
                sts = sts[sel]
                ags = ags[sel]
                sel = np.all(~np.isnan(wdata[sts, ags]), axis=-1)
                if np.any(sel):
                    sts = sts[sel]
                    ags = ags[sel]

                    dists = cdist(points[si, :, :2], wdata[sts, ags, :2])
                    j = np.argmin(dists, axis=1)
                    sts = sts[j]
                    ags = ags[j]
                    wake_si[si] = sts + wi0

                    nx = wdata[sts, ags, :2]
                    dp = points[si, :, :2] - nx
                    sel = ags < self.max_age - 1
                    if np.any(sel):
                        nx[sel] = wdata[sts[sel], ags[sel] + 1, :2] - nx[sel]
                    if np.any(~sel):
                        nx[~sel] -= wdata[sts[~sel], ags[~sel] - 1, :2]
                    dx = np.linalg.norm(nx, axis=-1)
                    nx /= dx[:, None] + 1e-14

                    projx = np.einsum("sd,sd->s", dp, nx)
                    sel = (projx > -dx) & (projx < dx)
                    if np.any(sel):
                        ny = np.concatenate([-nx[:, 1, None], nx[:, 0, None]], axis=1)

                        wcoos[si, sel, 0] = projx[sel] + wdata[sts[sel], ags[sel], 3]
                        wcoos[si, sel, 1] = np.einsum("sd,sd->s", dp[sel], ny[sel])

        # store turbines that cause wake:
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # store states that cause wake for each target point,
        # will be used by model.get_data() during wake calculation:
        tdata.add(
            FC.STATES_SEL,
            wake_si.reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )

        return wcoos.reshape(n_states, n_targets, n_tpoints, 3)
