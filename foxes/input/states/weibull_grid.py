import numpy as np
from os import PathLike
from xarray import Dataset, open_dataset

from foxes.core import States
from foxes.data import STATES
from foxes.utils import weibull_weights
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC


class WeibullGrid(States):
    """
    Weibull sectors at grid points

    Attributes
    ----------
    data_source: str or xarray.Dataset
        Either path to NetCDF file or data
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)
    var2ncvar: dict
        Mapping from foxes variable names to variable names
        in the nc file
    fixed_vars: dict
        Fixed uniform variable values, instead of
        reading from data
    interp_method: str
        The interpolation method, "linear" or "nearest"
    interp_pars: dict
        Additional arguments for the interpolation
    interp_fallback_nearest: bool
        If True, use nearest neighbor interpolation if the
        interpolation method fails.
    rpars: dict
        Additional parameters for reading the file
    RDICT: dict
        Default xarray file reading parameters

    :group: input.states

    """

    RDICT = {}

    def __init__(
        self,
        data_source,
        output_vars,
        ws_bins=None,
        var2ncvar={},
        fixed_vars={},
        sel=None,
        isel=None,
        interp_method="linear",
        interp_pars={},
        interp_fallback_nearest=False,
        read_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or xarray.Dataset
            Either path to NetCDF file path or data
        output_vars: list of str
            The output variables
        ws_bins: list of float, optional
            The wind speed bins, including
            lower and upper bounds
        var2ncvar: dict
            Mapping from foxes variable names to variable names
            in the nc file
        fixed_vars: dict
            Fixed uniform variable values, instead of
            reading from data
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        interp_method: str
            The interpolation method, "linear", "nearest" or "radialBasisFunction"
        interp_fallback_nearest: bool
            If True, use nearest neighbor interpolation if the
            interpolation method fails.
        interp_pars: dict
            Additional arguments for the interpolation
        read_pars: dict
            Additional parameters for reading the file
        kwargs: dict, optional
            Additional arguments for the base class

        """
        super().__init__(**kwargs)
        self.data_source = data_source
        self.ws_bins = None if ws_bins is None else np.asarray(ws_bins)
        self.var2ncvar = var2ncvar
        self.fixed_vars = fixed_vars
        self.sel = sel if sel is not None else {}
        self.isel = isel if isel is not None else {}
        self.rpars = read_pars
        self.interp_method = interp_method
        self.interp_pars = interp_pars
        self.interp_fallback_nearest = interp_fallback_nearest

        self._original_data = None
        self._ovars = output_vars
        self._heights = [100.0]
        self._n_wd = None
        self._n_ws = None
        self._N = None

        if FV.WS not in self._ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WS}', got {self._ovars}"
            )
        if FV.WD not in self._ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WD}', got {self._ovars}"
            )
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self._ovars:
                raise ValueError(
                    f"States '{self.name}': Cannot have '{v}' as output variable"
                )

    def __repr__(self):
        return f"{type(self).__name__}(n_pt={self._n_pt}, n_wd={self._n_wd}, n_ws={self._n_ws})"

    def _read_nc(self, algo, verbosity=0):
        """
        Extracts data from file or Dataset.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        data: xarray.Dataset
            The input data

        """
        # store original data
        if self._original_data is not None:
            self._data = self._original_data
            self._original_data = None

        # read file or grab data
        if isinstance(self.data_source, (str, PathLike)):
            fpath = get_input_path(self.data_source)
            if not fpath.is_file():
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Reading static data '{fpath}' from context '{STATES}'"
                    )
                fpath = algo.dbook.get_file_path(STATES, fpath.name, check_raw=False)
                if verbosity > 0:
                    print(f"Path: {fpath}")
            elif verbosity > 0:
                print(f"States '{self.name}': Reading file {fpath}")
            rpars = dict(self.RDICT, **self.rpars)
            data = open_dataset(fpath, engine=config.nc_engine, **rpars)
            self._original_data = data

        elif isinstance(self.data_source, Dataset):
            data = self.data_source

        else:
            raise TypeError(
                f"States '{self.name}': Expecting data_source to be a string or xarray.Dataset, got {type(self.data_source)}"
            )

        # optionally select a subset
        if self.isel is not None and len(self.isel):
            data = data.isel(**self.isel)
        if self.sel is not None and len(self.sel):
            data = data.sel(**self.sel)

        # remove wd 360 from the end, if wd 0 is given:
        cwd = self.var2ncvar.get(FV.WD, FV.WD)
        wd = data[cwd].to_numpy()
        if wd[0] == 0.0 and wd[-1] == 360.0:
            data = data.isel({cwd: np.s_[:-1]})

        # construct wind speed bins and bin deltas
        cws = self.var2ncvar.get(FV.WS, FV.WS)
        if self.ws_bins is not None:
            wsb = self.ws_bins
            wss = 0.5 * (wsb[:-1] + wsb[1:])
        elif cws in data:
            wss = data[cws].to_numpy()
            wsb = np.zeros((len(wss) + 1,), dtype=config.dtype_double)
            wsb[1:-1] = 0.5 * (wss[1:] + wss[:-1])
            wsb[0] = wss[0] - 0.5 * wsb[1]
            wsb[-1] = wss[-1] + 0.5 * wsb[-2]
            self.ws_bins = wsb
        else:
            raise ValueError(
                f"States '{self.name}': Expecting ws_bins argument, since '{cws}' not found in data"
            )
        wsd = wsb[1:] - wsb[:-1]
        n_ws = len(wss)
        del wsb

        # prepare data binning
        if self._original_data is None:
            self._original_data = self.data_source
        cx = self.var2ncvar.get(FV.X, FV.X)
        cy = self.var2ncvar.get(FV.Y, FV.Y)
        ch = self.var2ncvar.get(FV.H, FV.H)
        n_x = data.sizes[cx]
        n_y = data.sizes[cy]
        n_h = data.sizes[ch] if ch in data else 1
        n_wd = data.sizes[cwd]
        self.BIN_WD = self.var("bin_wd")
        self.BIN_WS = self.var("bin_ws")
        self.X = self.var(FV.X)
        self.Y = self.var(FV.Y)
        self.H = self.var(FV.H)
        shp_ds = (n_wd, n_ws)
        shp_h = (n_h,)
        dms_h = (self.H,)
        shp_xy = (n_x, n_y)
        dms_xy = (self.X, self.Y)
        shp_xyh = (n_x, n_y, n_h)
        dms_xyh = (self.X, self.Y, self.H)
        dms_ds = (self.BIN_WD, self.BIN_WS)
        shp_dsxy = (n_wd, n_ws, n_x, n_y)
        dms_dsxy = (self.BIN_WD, self.BIN_WS, self.X, self.Y)
        shp_dsxyh = (n_wd, n_ws, n_x, n_y, n_h)
        dms_dsxyh = (self.BIN_WD, self.BIN_WS, self.X, self.Y, self.H)

        # create binned data
        self._data = {
            FV.WD: np.zeros(shp_ds, dtype=config.dtype_double),
            FV.WS: np.zeros(shp_ds, dtype=config.dtype_double),
        }
        self._data[FV.WD][:] = data[cwd].to_numpy()[:, None]
        self._data[FV.WS][:] = wss[None, :]
        self._data[FV.WD] = (dms_ds, self._data[FV.WD])
        self._data[FV.WS] = (dms_ds, self._data[FV.WS])
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT] + self._ovars:
            if v not in [FV.WS, FV.WD] and v not in self.fixed_vars:
                w = self.var2ncvar.get(v, v)
                if w not in data:
                    raise KeyError(
                        f"States '{self.name}': Missing variable '{w}' in data, found {list(data.data_vars.keys())}"
                    )
                if v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
                    if cws in data[w].dims:
                        raise ValueError(
                            f"States '{self.name}': Cannot have '{cws}' as dimension in variable '{v}', got {data[w].dims}"
                        )
                    if cwd not in data[w].dims:
                        raise ValueError(
                            f"States '{self.name}': Expecting '{cwd}' as dimension in variable '{v}', got {data[w].dims}"
                        )
                    
                d = data[w]
                if d.dims == (cwd, cws):
                    self._data[v] = (dms_ds, d.to_numpy())

                elif d.dims == (cws, cwd):
                    self._data[v] = (dms_ds, np.swapaxes(d.to_numpy(), 0, 1))

                elif d.dims == (cwd,):
                    self._data[v] = np.zeros(shp_ds, dtype=config.dtype_double)
                    self._data[v][:] = d.to_numpy()[:, None]
                    self._data[v] = (dms_ds, self._data[v])

                elif d.dims == (cws,):
                    self._data[v] = np.zeros(shp_ds, dtype=config.dtype_double)
                    self._data[v][:] = d.to_numpy()[None, :]
                    self._data[v] = (dms_ds, self._data[v])

                elif len(d.dims) == 5 and sorted(d.dims) == sorted((cwd, cws, cx, cy, ch)):
                    ids = [d.dims.index(dms) for dms in (cwd, cws, cx, cy, ch)]
                    self._data[v] = (dms_dsxyh, np.moveaxis(d.to_numpy(), ids, [0, 1, 2, 3, 4]))

                elif len(d.dims) == 4 and sorted(d.dims) == sorted((cwd, cws, cx, cy)):
                    ids = [d.dims.index(dms) for dms in (cwd, cws, cx, cy)]
                    self._data[v] = (dms_dsxy, np.moveaxis(d.to_numpy(), ids, [0, 1, 2, 3]))

                elif len(d.dims) == 4 and sorted(d.dims) == sorted((cwd, cx, cy, ch)):
                    ids = [d.dims.index(dms) for dms in (cwd, cx, cy, ch)]
                    self._data[v] = np.zeros(shp_dsxyh, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d.to_numpy(), ids, [0, 1, 2, 3])[:, None, :, :, :]
                    self._data[v] = (dms_dsxyh, self._data[v])  

                elif len(d.dims) == 4 and sorted(d.dims) == sorted((cws, cx, cy, ch)):
                    ids = [d.dims.index(dms) for dms in (cws, cx, cy, ch)]
                    self._data[v] = np.zeros(shp_dsxyh, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d.to_numpy(), ids, [0, 1, 2, 3])[None, :, :, :, :]
                    self._data[v] = (dms_dsxyh, self._data[v])  

                elif len(d.dims) == 3 and sorted(d.dims) == sorted((cwd, cx, cy)):
                    ids = [d.dims.index(dms) for dms in (cwd, cx, cy)]
                    self._data[v] = np.zeros(shp_dsxy, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d.to_numpy(), ids, [0, 1, 2])[:, None, :, :]
                    self._data[v] = (dms_dsxy, self._data[v])

                elif len(d.dims) == 3 and sorted(d.dims) == sorted((cws, cx, cy)):
                    ids = [d.dims.index(dms) for dms in (cws, cx, cy)]
                    self._data[v] = np.zeros(shp_dsxy, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d.to_numpy(), ids, [0, 1, 2])[None, :, :, :]
                    self._data[v] = (dms_dsxy, self._data[v])

                elif len(d.dims) == 3 and sorted(d.dims) == sorted((cx, cy, ch)):
                    ids = [d.dims.index(dms) for dms in (cx, cy, ch)]
                    self._data[v] = (dms_xyh, np.moveaxis(d.to_numpy(), ids, [0, 1, 2]))

                elif len(d.dims) == 2 and sorted(d.dims) == sorted((cx, cy)):
                    ids = [d.dims.index(dms) for dms in (cx, cy)]
                    self._data[v] = (dms_xy, np.moveaxis(d.to_numpy(), ids, [0, 1]))

                elif d.dims == (ch,):
                    self._data[v] = (dms_h, d.to_numpy())

                else:
                    raise ValueError(f"States '{self.name}': Failed to map variable '{v}' with dimensions {d.dims} to expected dimensions {dms_ds} or {dms_dsxyh} or {dms_xyh}")

        # compute Weibull weights
        print("HERE", self._data[FV.WEIBULL_k][0], self._data[FV.WEIBULL_k][1].shape)
        s_ws = tuple([None, np.s_[:], None, None, None])
        s_wd = tuple([np.s_[:], np.s_[:], None, None, None])
        s_A = tuple([np.s_[:], np.s_[:]] + [np.s_[:] if x in self._data[FV.WEIBULL_A][0] else None for x in dms_xyh])
        s_k = tuple([np.s_[:], np.s_[:]] + [np.s_[:] if x in self._data[FV.WEIBULL_k][0] else None for x in dms_xyh])
        print(self._data[FV.WEIGHT][1].shape)
        print(wss[s_ws].shape)
        print(self._data[FV.WD][1][s_wd].shape)
        self._data[FV.WEIGHT][1] *= weibull_weights(
            ws=wss[s_ws],
            ws_deltas=wsd[s_wd],
            A=self._data.pop(FV.WEIBULL_A)[1][s_A],
            k=self._data.pop(FV.WEIBULL_k)[1][s_k],
        )

        # translate binned data to states
        self._n_pt = n_pt
        self._n_wd = n_wd
        self._n_ws = n_ws
        self._N = n_wd * n_ws
        self._data = Dataset(
            data_vars={v: (dms, d) for v, d in self._data.items()},
        )

        return data
    
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
        # read Weibull data from Dataset
        data = self._read_nc(algo, verbosity=0)
        print("HERE LDATA", data)
        quit()

        # read points
        cx = self.var2ncvar.get(FV.X, FV.X)
        cy = self.var2ncvar.get(FV.Y, FV.Y)
        ch = self.var2ncvar.get(FV.H, FV.H)
        self._points = [
            data[cx].to_numpy(),
            data[cy].to_numpy(),
        ]
        if ch in data:
            hts = data[ch].to_numpy()
            self._heights = np.unique(hts)
            if len(self._heights) > 1:
                self._points.append(hts)
            del hts
        self._points = np.stack(self._points, axis=-1)
        del data

        # translate binned data to states
        self.DATA = self.var("data")
        self.VARS = self.var("vars")
        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = []
        data = []
        for v, d in self._data.items():
            idata["coords"][self.VARS].append(v)
            data.append(d.to_numpy().reshape(self._N, self._n_pt))
        idata["data_vars"][self.DATA] = (
            (FC.STATE, self.POINT, self.VARS),
            np.stack(data, axis=-1),
        )
        self._data = None

        return idata

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
        return self._ovars

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

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

        # prepare interpolation data

        n_states, n_dpts, n_vars = mdata[self.DATA].shape
        n_pts = n_states * tdata.n_targets * tdata.n_tpoints
        n_dms = self._points.shape[1]

        qts = np.zeros((n_dpts, n_states, n_dms + 1), dtype=config.dtype_double)
        qts[..., :n_dms] = self._points[:, None, :]
        qts[..., n_dms] = np.arange(n_states)[
            None, :
        ]  # last dimension is the state index
        n_qts = n_dpts * n_states
        qts = qts.reshape(n_qts, n_dms + 1)

        data = np.swapaxes(
            mdata[self.DATA], 0, 1
        )  # now shape (n_dpts, n_states, n_vars)
        data = data.reshape(n_qts, n_vars)

        pts = np.zeros(
            (n_states, tdata.n_targets, tdata.n_tpoints, n_dms + 1),
            dtype=config.dtype_double,
        )
        pts[..., :n_dms] = tdata[FC.TARGETS][..., :n_dms]
        pts[..., n_dms] = np.arange(n_states)[
            :, None, None
        ]  # last dimension is the state index
        pts = pts.reshape(n_pts, n_dms + 1)

        # create interpolator
        if self.interp_method == "linear":
            interp = LinearNDInterpolator(qts, data, **self.interp_pars)
        elif self.interp_method == "nearest":
            interp = NearestNDInterpolator(qts, data, **self.interp_pars)
        elif self.interp_method == "radialBasisFunction":
            pars = {"neighbors": 10}
            pars.update(self.interp_pars)
            interp = RBFInterpolator(qts, data, **pars)
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
                interp = NearestNDInterpolator(qts, data)
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
        del pts, qts, data

        # prepare results
        ires = ires.reshape(n_states, tdata.n_targets, tdata.n_tpoints, n_vars)
        results = {str(v): ires[..., i] for i, v in enumerate(mdata[self.VARS])}
        results.update(
            {v: np.full_like(ires[..., 0], d) for v, d in self.fixed_vars.items()}
        )

        return results
