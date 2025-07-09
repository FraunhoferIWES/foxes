import numpy as np
from os import PathLike
from xarray import Dataset, open_dataset

from foxes.data import STATES
from foxes.utils import weibull_weights
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC

from .field_data import FieldData

class WeibullField(FieldData):
    """
    Weibull sectors at regular grid points

    Attributes
    ----------
    data_source: str or xarray.Dataset
        Either path to NetCDF file or data
    wd_coord: str
        The wind direction coordinate name
    ws_coord: str
        The wind speed coordinate name, if wind speed bin
        centres are in data, else None
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

    def __init__(
        self,
        data_source,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        wd_coord="wd",
        ws_coord=None,
        ws_bins=None,
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
        var2ncvar: dict, optional
            Mapping from variable names to variable names
            in the nc file
        fixed_vars: dict, optional
            Uniform values for output variables, instead
            of reading from data
        wd_coord: str
            The wind direction coordinate name
        ws_coord: str, optional
            The wind speed coordinate name, if wind speed bin
            centres are in data
        ws_bins: list of float, optional
            The wind speed bins, including
            lower and upper bounds
        kwargs: dict, optional
            Additional arguments for the base class

        """
        super().__init__(
            data_source=data_source,
            output_vars=output_vars,
            var2ncvar=var2ncvar,
            fixed_vars=fixed_vars,
            states_coord=wd_coord, 
            time_format=None, 
            **kwargs,
        )
        self.ws_bins = None if ws_bins is None else np.sort(np.asarray(ws_bins))
        self.ws_coord = ws_coord

        assert ws_coord is None or ws_bins is None, (
            f"States '{self.name}': Cannot have both ws_coord '{ws_coord}' and ws_bins {ws_bins}"
        )
        assert ws_coord is not None or ws_bins is not None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins"
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

    def _read_ds(self, ds, variables, verbosity=0):
        """
        Extract data from the original Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent
        
        Returns
        -------
        s: numpy.ndarray
            The states coordinate values
        x: numpy.ndarray
            The x coordinate values
        y: numpy.ndarray
            The y coordinate values
        h: numpy.ndarray or None
            The height coordinate values, or None if not available
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        # check for ws_coord
        if self.ws_coord is not None:
            assert self.ws_coord in ds.coords, (
                f"States '{self.name}': Expecting ws_coord '{self.ws_coord}' to be among data coordinates, got {list(ds.coords.keys())}"
            )
            for v, d in ds.data_vars.items():
                if self.ws_coord in d.dims:
                    raise NotImplementedError(
                        f"States '{self.name}': Cannot handle variable '{v}' with dimension '{self.ws_coord}' in data, dims = {d.dims}"
                    )
        
        # read data, using wd_coord as state coordinate
        wd, x, y, h, data0 = super()._read_ds(
            ds, variables, verbosity)

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
        elif self.ws_coord in ds.coords:
            wss = ds[self.ws_coord].to_numpy()
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
        del wsb, ds

        # calculate Weibull weights
        dms = [FV.WS, FV.WD]
        shp = [n_ws, n_wd]
        for c in [FV.X, FV.Y, FV.H]:
            for v in [FV.WEIBULL_A, FV.WEIBULL_k]:
                if c in data0[v][0]:
                    dms.append(c)
                    shp.append(data0[v][1].shape[data0[v][0].index(c)])
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
            w * weibull_weights(
                ws=wss[s_ws],
                ws_deltas=wsd[s_ws],
                A=data0.pop(FV.WEIBULL_A)[1][s_A],
                k=data0.pop(FV.WEIBULL_k)[1][s_k],
            )
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
            if dims[0] == FV.WD:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws] + list(d.shape)
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            elif len(dims) >=2 and dims[:2] == (FV.WS, FV.WD):
                dms = tuple([FC.STATE] + list(dims[2:]))
                shp = [self._N] + list(d.shape[2:])
                data[v] = (dms, d.reshape(shp))
            else:
                data[v] = (dims, d)
        data0 = data

        return None, x, y, h, data

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
        return self.ovars

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N
