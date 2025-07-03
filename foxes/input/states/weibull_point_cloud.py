import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from foxes.core import States
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .weibull_sectors import WeibullSectors


class WeibullPointCloud(States):
    """
    Weibull sectors at point cloud support, e.g., at turbine locations.

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
        self._n_pt = None
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
        data = WeibullSectors._read_nc(self, algo, point_coord=FC.POINT, verbosity=0)

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
