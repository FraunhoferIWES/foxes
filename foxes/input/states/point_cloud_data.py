import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from foxes.config import config
from foxes.utils import wd2uv, uv2wd, weibull_weights
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class PointCloudData(DatasetStates):
    """
    Inflow data with point cloud support.

    Attributes
    ----------
    states_coord: str
        The states coordinate name in the data
    point_coord: str
        The point coordinate name in the data
    x_ncvar: str
        The x variable name in the data
    y_ncvar: str
        The y variable name in the data
    h_ncvar: str, optional
        The height variable name in the data
    weight_ncvar: str, optional
        The name of the weights variable in the data
    interp_method: str
        The interpolation method, "linear", "nearest" or "radialBasisFunction"
    interp_fallback_nearest: bool
        If True, use nearest neighbor interpolation if the
        interpolation method fails.
    interp_pars: dict
        Additional arguments for the interpolation

    :group: input.states

    """

    def __init__(
        self,
        *args,
        states_coord="Time",
        point_coord="point",
        x_ncvar="x",
        y_ncvar="y",
        h_ncvar=None,
        weight_ncvar=None,
        interp_method="linear",
        interp_fallback_nearest=False,
        interp_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        states_coord: str
            The states coordinate name in the data
        point_coord: str
            The point coordinate name in the data
        x_ncvar: str
            The x variable name in the data
        y_ncvar: str
            The y variable name in the data
        h_ncvar: str, optional
            The height variable name in the data
        weight_ncvar: str, optional
            The name of the weights variable in the data
        interp_method: str
            The interpolation method, "linear", "nearest" or "radialBasisFunction"
        interp_fallback_nearest: bool
            If True, use nearest neighbor interpolation if the
            interpolation method fails.
        interp_pars: dict
            Additional arguments for the interpolation
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        self.states_coord = states_coord
        self.point_coord = point_coord
        self.x_ncvar = x_ncvar
        self.y_ncvar = y_ncvar
        self.h_ncvar = h_ncvar
        self.weight_ncvar = weight_ncvar
        self.interp_method = interp_method
        self.interp_pars = interp_pars
        self.interp_fallback_nearest = interp_fallback_nearest

        self.variables = [FV.X, FV.Y]
        self.variables += [v for v in self.ovars if v not in self.fixed_vars]
        self.var2ncvar[FV.X] = x_ncvar
        self.var2ncvar[FV.Y] = y_ncvar
        if weight_ncvar is not None:
            self.var2ncvar[FV.WEIGHT] = weight_ncvar
            self.variables.append(FV.WEIGHT)
        elif FV.WEIGHT in self.var2ncvar:
            raise KeyError(
                f"States '{self.name}': Cannot have '{FV.WEIGHT}' in var2ncvar, use weight_ncvar instead"
            )

        self._n_pt = None
        self._n_wd = None
        self._n_ws = None

        if FV.WS not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WS}', got {self.ovars}"
            )
        if FV.WD not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WD}', got {self.ovars}"
            )
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self.ovars:
                raise ValueError(
                    f"States '{self.name}': Cannot have '{v}' as output variable"
                )

        self._cmap = {
            FC.STATE: self.states_coord,
            FC.POINT: self.point_coord,
        }

    def __repr__(self):
        return f"{type(self).__name__}(n_pt={self._n_pt}, n_wd={self._n_wd}, n_ws={self._n_ws})"

    def _read_ds(self, ds, cmap, variables, verbosity=0):
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        coords: dict
            keys: Foxes variable names, values: 1D coordinate value arrays
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        coords, data = super()._read_ds(ds, cmap, variables, verbosity)

        assert FV.X in data and FV.Y in data, (
            f"States '{self.name}': Expecting variables '{FV.X}' and '{FV.Y}' in data, found {list(data.keys())}"
        )
        assert data[FV.X][0] == (FC.POINT,), (
            f"States '{self.name}': Expecting variable '{FV.X}' to have dimensions '({FC.POINT},)', got {data[FV.X][0]}"
        )
        assert data[FV.Y][0] == (FC.POINT,), (
            f"States '{self.name}': Expecting variable '{FV.Y}' to have dimensions '({FC.POINT},)', got {data[FV.Y][0]}"
        )
        if FV.H in data:
            assert data[FV.H][0] == (FC.POINT,), (
                f"States '{self.name}': Expecting variable '{FV.H}' to have dimensions '({FC.POINT},)', got {data[FV.H][0]}"
            )

        points = [data.pop(FV.X)[1], data.pop(FV.Y)[1]]
        if FV.H in data:
            points.append(data.pop(FV.H)[1])
        coords[FC.POINT] = np.stack(points, axis=-1)

        return coords, data

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
        return super().load_data(
            algo,
            cmap=self._cmap,
            variables=self.variables,
            bounds_extra_space=None,
            verbosity=verbosity,
        )

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
        # prepare
        self.ensure_output_vars(algo, tdata)
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        n_states = fdata.n_states
        n_pts = n_states * n_targets * n_tpoints
        coords = [self.states_coord, self.point_coord]

        # get data for calculation
        coords, data, weights = self.get_calc_data(mdata, self._cmap, self.variables)
        coords[FC.STATE] = np.arange(n_states, dtype=config.dtype_int)

        # interpolate data to points:
        out = {}
        for dims, (vrs, d) in data.items():
            # prepare
            n_vrs = len(vrs)
            qts = coords[FC.POINT]
            n_qts, n_dms = qts.shape
            idims = dims[:-1]

            if idims == (FC.STATE,):
                for i, v in enumerate(vrs):
                    if v in self.ovars:
                        out[v] = np.zeros(
                            (n_states, n_targets, n_tpoints), dtype=config.dtype_double
                        )
                        out[v][:] = d[:, None, None, i]
                continue

            elif idims == (FC.POINT,):
                # prepare grid data
                gts = qts
                n_gts = n_qts

                # prepare evaluation points
                pts = tdata[FC.TARGETS][..., :n_dms].reshape(n_pts, n_dms)

            elif idims == (FC.STATE, FC.POINT):
                # prepare grid data, add state index to last axis
                gts = np.zeros((n_qts, n_states, n_dms + 1), dtype=config.dtype_double)
                gts[..., :n_dms] = qts[:, None, :]
                gts[..., n_dms] = np.arange(n_states)[None, :]
                n_gts = n_qts * n_states
                gts = gts.reshape(n_gts, n_dms + 1)

                # reorder data, first to shape (n_qts, n_states, n_vars),
                # then to (n_gts, n_vrs)
                d = np.swapaxes(d, 0, 1)
                d = d.reshape(n_gts, n_vrs)

                # prepare evaluation points, add state index to last axis
                pts = np.zeros(
                    (n_states, tdata.n_targets, tdata.n_tpoints, n_dms + 1),
                    dtype=config.dtype_double,
                )
                pts[..., :n_dms] = tdata[FC.TARGETS][..., :n_dms]
                pts[..., n_dms] = np.arange(n_states)[:, None, None]
                pts = pts.reshape(n_pts, n_dms + 1)

            else:
                raise ValueError(
                    f"States '{self.name}': Unsupported dimensions {dims} for variables {vrs}"
                )

            # translate (WD, WS) to (U, V):
            if FV.WD in vrs or FV.WS in vrs:
                assert FV.WD in vrs and (FV.WS in vrs or FV.WS in self.fixed_vars), (
                    f"States '{self.name}': Missing '{FV.WD}' or '{FV.WS}' in data variables {vrs} for dimensions {dims}"
                )
                iwd = vrs.index(FV.WD)
                iws = vrs.index(FV.WS)
                ws = d[..., iws] if FV.WS in vrs else self.fixed_vars[FV.WS]
                d[..., [iwd, iws]] = wd2uv(d[..., iwd], ws, axis=-1)
                del ws

            # create interpolator
            if self.interp_method == "linear":
                interp = LinearNDInterpolator(gts, d, **self.interp_pars)
            elif self.interp_method == "nearest":
                interp = NearestNDInterpolator(gts, d, **self.interp_pars)
            elif self.interp_method == "radialBasisFunction":
                pars = {"neighbors": 10}
                pars.update(self.interp_pars)
                interp = RBFInterpolator(gts, d, **pars)
            else:
                raise NotImplementedError(
                    f"States '{self.name}': Interpolation method '{self.interp_method}' not implemented, choices are: 'linear', 'nearest', 'radialBasisFunction'"
                )

            # run interpolation
            ires = interp(pts)
            del interp

            # check for error:
            sel = np.any(np.isnan(ires), axis=-1)
            if np.any(sel):
                i = np.where(sel)[0]
                if self.interp_fallback_nearest:
                    interp = NearestNDInterpolator(gts, d)
                    pts = pts[i]
                    ires[i] = interp(pts)
                    del interp
                else:
                    p = pts[i[0], :n_dms]
                    qmin = np.min(qts[:, :n_dms], axis=0)
                    qmax = np.max(qts[:, :n_dms], axis=0)
                    raise ValueError(
                        f"States '{self.name}': Interpolation method '{self.interp_method}' failed for {np.sum(sel)} points, e.g. for point {p}, outside of bounds {qmin} - {qmax}"
                    )
            del pts, gts, d

            # translate (U, V) into (WD, WS):
            if FV.WD in vrs:
                uv = ires[..., [iwd, iws]]
                ires[..., iwd] = uv2wd(uv)
                ires[..., iws] = np.linalg.norm(uv, axis=-1)
                del uv

            # set output:
            for i, v in enumerate(vrs):
                out[v] = ires[..., i].reshape(n_states, n_targets, n_tpoints)
            del ires

        # set fixed variables:
        for v, d in self.fixed_vars.items():
            out[v] = np.full(
                (n_states, n_targets, n_tpoints), d, dtype=config.dtype_double
            )

        # add weights:
        if weights is not None:
            tdata[FV.WEIGHT] = weights[:, None, None]
        elif FV.WEIGHT in out:
            tdata[FV.WEIGHT] = out.pop(FV.WEIGHT)
        else:
            tdata[FV.WEIGHT] = np.full(
                (n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
            )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: out[v] for v in self.ovars}


class WeibullPointCloud(PointCloudData):
    """
    Weibull sectors at point cloud support, e.g., at turbine locations.

    Attributes
    ----------
    wd_coord: str
        The wind direction coordinate name
    ws_coord: str
        The wind speed coordinate name, if wind speed bin
        centres are in data, else None
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)

    :group: input.states

    """

    def __init__(
        self,
        *args,
        wd_coord,
        ws_coord=None,
        ws_bins=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Positional arguments for the base class
        wd_coord: str
            The wind direction coordinate name
        ws_coord: str, optional
            The wind speed coordinate name, if wind speed bin
            centres are in data
        ws_bins: list of float, optional
            The wind speed bins, including
            lower and upper bounds
        kwargs: dict, optional
            Keyword arguments for the base class

        """
        super().__init__(
            *args,
            states_coord=wd_coord,
            time_format=None,
            load_mode="preload",
            **kwargs,
        )
        self.wd_coord = wd_coord
        self.ws_coord = ws_coord
        self.ws_bins = None if ws_bins is None else np.sort(np.asarray(ws_bins))

        assert ws_coord is not None or ws_bins is not None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins"
        )
        assert ws_coord is None or ws_bins is None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins, not both"
        )

        if FV.WD not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WD}', got {self.ovars}"
            )
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self.ovars:
                raise ValueError(
                    f"States '{self.name}': Cannot have '{v}' as output variable"
                )
            if v not in self.variables:
                self.variables.append(v)

        for v in [FV.WS, FV.WD]:
            if v in self.variables:
                self.variables.remove(v)

        self._n_wd = None
        self._n_ws = None

    def __repr__(self):
        return f"{type(self).__name__}(n_wd={self._n_wd}, n_ws={self._n_ws})"

    def _read_ds(self, ds, cmap, variables, verbosity=0):
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        coords: dict
            keys: Foxes variable names, values: 1D coordinate value arrays
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        # read data, using wd_coord as state coordinate
        hcmap = cmap.copy()
        if self.ws_coord is not None:
            hcmap = {FV.WS: self.ws_coord, **cmap}
        coords, data0 = super()._read_ds(ds, hcmap, variables, verbosity)
        wd = coords.pop(FC.STATE)
        wss = coords.pop(FV.WS, None)

        # replace state by wd coordinate
        data0 = {
            v: (tuple({FC.STATE: FV.WD}.get(c, c) for c in dims), d)
            for v, (dims, d) in data0.items()
        }

        # check weights
        if FV.WEIGHT not in data0:
            raise KeyError(
                f"States '{self.name}': Missing weights variable '{FV.WEIGHT}' in data, found {sorted(list(data0.keys()))}"
            )
        else:
            dims = data0[FV.WEIGHT][0]
            if FV.WD not in dims:
                raise KeyError(
                    f"States '{self.name}': Expecting weights variable '{FV.WEIGHT}' to contain dimension '{FV.WD}', got {dims}"
                )
            if FV.WS in dims:
                raise KeyError(
                    f"States '{self.name}': Expecting weights variable '{FV.WEIGHT}' to not contain dimension '{FV.WS}', got {dims}"
                )

        # construct wind speed bins and bin deltas
        assert FV.WS not in data0, (
            f"States '{self.name}': Cannot have '{FV.WS}' in data, found variables {list(data0.keys())}"
        )
        if self.ws_bins is not None:
            wsb = self.ws_bins
            wss = 0.5 * (wsb[:-1] + wsb[1:])
        elif wss is not None:
            wsb = np.zeros((len(wss) + 1,), dtype=config.dtype_double)
            wsb[1:-1] = 0.5 * (wss[1:] + wss[:-1])
            wsb[0] = wss[0] - 0.5 * wsb[1]
            wsb[-1] = wss[-1] + 0.5 * wsb[-2]
            self.ws_bins = wsb
        else:
            raise ValueError(
                f"States '{self.name}': Expecting ws_bins argument, or '{self.ws_coord}' among data coordinates, got {list(ds.coords.keys())}"
            )
        wsd = wsb[1:] - wsb[:-1]
        n_ws = len(wss)
        n_wd = len(wd)
        del wsb

        # calculate Weibull weights
        dms = [FV.WS, FV.WD]
        shp = [n_ws, n_wd]
        for v in [FV.WEIBULL_A, FV.WEIBULL_k]:
            if FC.POINT in data0[v][0]:
                dms.append(FC.POINT)
                shp.append(data0[v][1].shape[data0[v][0].index(FC.POINT)])
                break
        dms = tuple(dms)
        shp = tuple(shp)
        if data0[FV.WEIGHT][0] == dms:
            w = data0.pop(FV.WEIGHT)[1]
        else:
            s_w = tuple([np.s_[:] if c in data0[FV.WEIGHT][0] else None for c in dms])
            w = np.zeros(shp, dtype=config.dtype_double)
            w[:] = data0.pop(FV.WEIGHT)[1][s_w]
        s_ws = tuple([np.s_[:], None] + [None] * (len(dms) - 2))
        s_A = tuple([np.s_[:] if c in data0[FV.WEIBULL_A][0] else None for c in dms])
        s_k = tuple([np.s_[:] if c in data0[FV.WEIBULL_A][0] else None for c in dms])
        data0[FV.WEIGHT] = (
            dms,
            w
            * weibull_weights(
                ws=wss[s_ws],
                ws_deltas=wsd[s_ws],
                A=data0.pop(FV.WEIBULL_A)[1][s_A],
                k=data0.pop(FV.WEIBULL_k)[1][s_k],
            ),
        )
        del w, s_ws, s_A, s_k

        # translate binned data to states
        self._N = n_ws * n_wd
        self._inds = None
        data = {
            FV.WS: np.zeros((n_ws, n_wd), dtype=config.dtype_double),
            FV.WD: np.zeros((n_ws, n_wd), dtype=config.dtype_double),
        }
        data[FV.WS][:] = wss[:, None]
        data[FV.WD][:] = wd[None, :]
        data[FV.WS] = ((FC.STATE,), data[FV.WS].reshape(self._N))
        data[FV.WD] = ((FC.STATE,), data[FV.WD].reshape(self._N))
        for v in list(data0.keys()):
            dims, d = data0.pop(v)
            if len(dims) >= 2 and dims[:2] == (FV.WS, FV.WD):
                dms = tuple([FC.STATE] + list(dims[2:]))
                shp = [self._N] + list(d.shape[2:])
                data[v] = (dms, d.reshape(shp))
            elif dims[0] == FV.WD:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws] + list(d.shape)
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            elif dims[0] == FV.WS:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws, n_wd] + list(d.shape[2:])
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[:, None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            else:
                data[v] = (dims, d)

        return coords, data
