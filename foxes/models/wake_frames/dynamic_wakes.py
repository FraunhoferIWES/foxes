import numpy as np

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

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if not isinstance(algo, Iterative):
            raise TypeError(
                f"Incompatible algorithm type {type(algo).__name__}, expecting {Iterative.__name__}"
            )
        super().initialize(algo, verbosity, force=force)

        # disable subset state selection in iterative algo:
        algo.conv_crit.disable_subsets()

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
        istates = mdata.chunki_states
        itargets = mdata.chunki_points
        prev_t = itargets
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + n_states
        dt = self._dt[i0:i1]
        tdi = {
            v: (FC.STATE, FC.TARGET, FC.TPOINT)
            for v in algo.states.output_point_vars(algo)
        }
        key = f"{self.DATA}_{downwind_index}"
        self.AGE = self.var("age")
        self.XYHL = self.var("xyhl")

        def ukey_fun(fr, to):
            """helper function to create update key"""
            return f"{self.UPDATE}_dw{downwind_index}_from_{fr}_to_{to}"

        # compute wakes that start within this chunk: x, y, z, length
        data = algo.get_from_chunk_store(
            name=key, mdata=mdata, prev_t=prev_t, error=itargets > 0
        )
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
                    htdata = TData.from_points(
                        points=pts[:, None], data=tdt, dims=tdi, mdata=hmdata
                    )
                    hdt = dt[:, None]
                else:
                    s = np.s_[age:]
                    pts = pts[:-1]
                    hmdata = mdata.get_slice(FC.STATE, s)
                    hfdata = fdata.get_slice(FC.STATE, s)
                    htdt = {v: d[s] for v, d in tdt.items()}
                    htdata = TData.from_points(
                        points=pts[:, None], data=htdt, dims=tdi, mdata=hmdata
                    )
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
            algo.add_to_chunk_store(
                key,
                data,
                dims=(FC.STATE, self.AGE, self.XYHL),
                mdata=mdata,
                copy=False,
            )
            algo.block_convergence(mdata=mdata)

        # wake dynamics are computed during calc_farm, hence for itargets = 0 only:
        prev_s = 0
        wi0 = i0
        data = [data]
        if itargets == 0:
            # apply updates from future chunks:
            for (jstates, jtargets), cdict in algo.chunk_store.items():
                uname = ukey_fun(jstates, istates)
                if jstates > istates and jtargets == 0 and uname in cdict:
                    u = cdict[uname]
                    if u is not None:
                        sel = np.isnan(data[0]) & ~np.isnan(u)
                        if np.any(sel):
                            data[0][sel] = u[sel]
                            algo.add_to_chunk_store(
                                key,
                                data[0],
                                dims=(FC.STATE, self.AGE, self.XYHL),
                                mdata=mdata,
                                copy=False,
                            )
                            algo.block_convergence(mdata=mdata)
                        cdict[uname] = None
                        del sel
                    del u

            # compute wakes from previous chunks:
            while istates - prev_s > 0:
                prev_s += 1

                # read data from previous chunk:
                hdata, (h_i0, h_n_states, __, __) = algo.get_from_chunk_store(
                    name=key,
                    mdata=mdata,
                    prev_s=prev_s,
                    prev_t=prev_t,
                    ret_inds=True,
                    error=False,
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
                                v: np.zeros(
                                    (n_states, n_pts, 1), dtype=config.dtype_double
                                )
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
                                    points=pts[None, :],
                                    data=htdt,
                                    dims=tdi,
                                    mdata=hmdata,
                                )
                                hdt = dt[s, None]
                                del htdt, s

                                res = algo.states.calculate(
                                    algo, hmdata, hfdata, htdata
                                )
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

                                hsel = (h_i0 + sts + ags < i1) & (
                                    ags < self.max_age - 1
                                )
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
                                    ukey_fun(istates, istates - prev_s),
                                    udata,
                                    dims=(),
                                    mdata=mdata,
                                    copy=False,
                                )
                                algo.block_convergence(mdata=mdata)

                            del udata, tdt
                        del pts

                    # store prev_s chunk's results:
                    data.insert(0, hdata)

                    del sts, ags, sel
                del hdata

        # for itargets > 0, just gather previous computed data:
        else:
            while istates - prev_s > 0:
                prev_s += 1

                # read data from previous chunk:
                hdata, (h_i0, h_n_states, __, __) = algo.get_from_chunk_store(
                    name=key,
                    mdata=mdata,
                    prev_s=prev_s,
                    prev_t=prev_t,
                    ret_inds=True,
                    error=False,
                )
                if hdata is None:
                    break
                else:
                    data.insert(0, hdata)
                    wi0 = h_i0

        if len(data) == 1:
            data = data[0]
        else:
            data = np.concatenate(data, axis=0)

        """
        # wake path plot for debugging:
        if algo.final_iteration and itargets == 0:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            for i in range(data.shape[0]):
                ax.plot(data[i, :, 0], data[i, :, 1], label=f"state {i}")
            ax.set_title(f"{(mdata.chunki_states, mdata.chunki_points)}: {key} {data.shape}")
            #ax.legend()
            plt.show()
            plt.close(fig)
        """

        return data, wi0

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

        # loop over states:
        for si in range(n_states):
            # select wake ages that exist for this state:
            ags = np.arange(self.max_age)
            sts = i0 + si - ags - wi0
            sel = (sts >= 0) & (sts < len(wdata))
            if np.any(sel):
                # filter to existing wake points:
                sts = sts[sel]
                ags = ags[sel]
                done = np.zeros(len(ags), dtype=bool)
                for aprx in ["c", "f", "b", "o"]:
                    stsd = None
                    ags1 = None
                    ags0 = None
                    agsd = None

                    # first step:
                    if aprx == "o" and len(ags) == 1 and ags[0] == 0:
                        sel = ~done & ~np.isnan(wdata[sts, ags, 0])
                        if np.any(sel):
                            stsd = sts[sel]
                            ags0 = ags[sel]
                            agsd = ags0

                    # central:
                    elif aprx == "c":
                        ags0 = np.maximum(ags - 1, 0)
                        ags1 = np.minimum(ags + 1, wdata.shape[1] - 1)
                        sel = (
                            ~done
                            & (ags > 0)
                            & (ags < wdata.shape[1] - 1)
                            & ~np.isnan(wdata[sts, ags0, 0])
                            & ~np.isnan(wdata[sts, ags1, 0])
                        )
                        if np.any(sel):
                            stsd = sts[sel]
                            ags0 = ags[sel] - 1
                            ags1 = ags0 + 2
                            agsd = ags[sel]

                    # forward:
                    elif aprx == "f":
                        agsd = np.minimum(ags + 1, wdata.shape[1] - 1)
                        sel = (
                            ~done
                            & (ags < wdata.shape[1] - 1)
                            & ~np.isnan(wdata[sts, ags, 0])
                            & ~np.isnan(wdata[sts, agsd, 0])
                        )
                        if np.any(sel):
                            stsd = sts[sel]
                            ags0 = ags[sel]
                            ags1 = ags0 + 1
                            agsd = ags0

                    # backward:
                    elif aprx == "b":
                        agsd = np.maximum(ags - 1, 0)
                        sel = (
                            ~done
                            & (ags > 0)
                            & ~np.isnan(wdata[sts, agsd, 0])
                            & ~np.isnan(wdata[sts, ags, 0])
                        )
                        if np.any(sel):
                            stsd = sts[sel]
                            ags1 = ags[sel]
                            ags0 = ags1 - 1
                            agsd = ags1

                    if stsd is not None:
                        # single wake point case, must originate from rotor centre:
                        if aprx == "o":
                            nx = wd2uv(fdata[FV.WD][si, downwind_index])[None, :2]
                            dx = wdata[stsd, ags0, 3]

                        # compute wake tangent vectors, using next and current wake age points:
                        else:
                            nx = wdata[stsd, ags1, :2] - wdata[stsd, ags0, :2]
                            dx = np.linalg.norm(nx, axis=-1)
                            nx /= dx[:, None] + 1e-14

                        # project target points onto wake points:
                        dp = points[si, :, None, :2] - wdata[None, stsd, agsd, :2]
                        projx = (
                            dp[:, :, 0] * nx[None, :, 0] + dp[:, :, 1] * nx[None, :, 1]
                        )
                        projy = (
                            -dp[:, :, 0] * nx[None, :, 1] + dp[:, :, 1] * nx[None, :, 0]
                        )
                        selp = (
                            (projx > -dx[None, :])
                            & (projx < dx[None, :])
                            & (
                                np.isnan(wcoos[si, :, None, 1])
                                | (np.abs(projy) < np.abs(wcoos[si, :, None, 1]))
                            )
                        )
                        if np.any(selp):
                            w = np.where(selp)
                            wcoos[si, w[0], 0] = (
                                projx[selp] + wdata[stsd[w[1]], agsd[w[1]], 3]
                            )
                            wcoos[si, w[0], 1] = projy[selp]
                            del w

                        done[sel] = True
                        del dp, projx, projy, selp
                    del stsd, ags1, ags0, agsd

                    if np.all(done):
                        break

                del done
            del sts, ags, sel

        # store turbines that cause wake:
        tdata[FC.STATE_SOURCE_ORDERI] = downwind_index

        # store states that cause wake for each target point,
        # will be used by model.get_data() during wake calculation:
        tdata.add(
            FC.STATES_SEL,
            wake_si.reshape(n_states, n_targets, n_tpoints),
            (FC.STATE, FC.TARGET, FC.TPOINT),
        )

        return algo.wake_deflection.calc_deflection(
            algo,
            mdata,
            fdata,
            tdata,
            downwind_index,
            wcoos.reshape(n_states, n_targets, n_tpoints, 3),
        )
