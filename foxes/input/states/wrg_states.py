import numpy as np
from scipy.interpolate import interpn

from foxes.core.states import States
from foxes.config import config, get_input_path
from foxes.utils.wrg_utils import ReaderWRG
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC


class WRGStates(States):
    """
    Ambient states based on WRG data

    Attributes
    ----------
    wrg_fname: str
        Name of the WRG file
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)
    fixed_vars: dict
        Fixed uniform variable values, instead of
        reading from data
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    interpn_pars: dict
        Additional parameters for scipy.interpolate.interpn

    :group: input.states
    
    """

    def __init__(
            self, 
            wrg_fname, 
            ws_bins, 
            fixed_vars={},
            bounds_extra_space="1D",
            **interpn_pars,
        ):
        """
        Constructor
        
        Parameters
        ----------
        wrg_fname: str
            Name of the WRG file
        ws_bins: list of float
            The wind speed bins, including
            lower and upper bounds
        fixed_vars: dict
            Fixed uniform variable values, instead of
            reading from data
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        interpn_pars: dict, optional
            Additional parameters for scipy.interpolate.interpn

        """
        super().__init__()
        self.wrg_fname = wrg_fname
        self.ws_bins = np.asarray(ws_bins)
        self.fixed_vars = fixed_vars
        self.bounds_extra_space = bounds_extra_space
        self.interpn_pars = interpn_pars

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
        # read wrg file:
        fpath = get_input_path(self.wrg_fname)
        if not fpath.is_file():
            if verbosity > 0:
                print(
                    f"States '{self.name}': Reading static data '{self.wrg_fname}' from context '{STATES}'"
                )
            fpath = algo.dbook.get_file_path(
                STATES, self.wrg_fname, check_raw=False
            )
            if verbosity > 0:
                print(f"Path: {fpath}")
        elif verbosity:
            print(f"States '{self.name}': Reading file {fpath}")
        wrg = ReaderWRG(fpath)
        p0 = np.array([wrg.x0, wrg.y0], dtype=config.dtype_double)
        nx = wrg.nx
        ny = wrg.ny
        ns = wrg.n_sectors
        res = wrg.resolution

        # find bounds:
        if self.bounds_extra_space is not None:
            xy_min, xy_max = algo.farm.get_xy_bounds(
                extra_space=self.bounds_extra_space, algo=algo
            )
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting to bounds {xy_min} - {xy_max}"
                )
            ij_min = np.asarray((xy_min - p0)/res, dtype=config.dtype_int)
            ij_max = np.asarray((xy_max - p0)/res, dtype=config.dtype_int) + 1
            sx = slice(ij_min[0], ij_max[0])
            sy = slice(ij_min[1], ij_max[1])
        else:
            sx = np.s_[:]
            sy = np.s_[:]
        self._x = p0[0] + np.arange(nx) * res
        self._x = self._x[sx]
        self._y = p0[1] + np.arange(ny) * res
        self._y = self._y[sy]
        if len(self._x) < 2 or len(self._y) < 2:
            p1 = p0 + np.array([nx * res, ny * res])
            raise ValueError(f"No overlap with data at {p0} -- {p1}")

        # store data:
        A = []
        k = []
        f = []
        for s in range(ns):
            A.append(
                wrg.data[f"As_{s}"].to_numpy().reshape(ny, nx)[sy, sx]
            )
            k.append(
                wrg.data[f"Ks_{s}"].to_numpy().reshape(ny, nx)[sy, sx]
            )
            f.append(
                wrg.data[f"fs_{s}"].to_numpy().reshape(ny, nx)[sy, sx]
            )
        del wrg
        A = np.stack(A, axis=0).T
        k = np.stack(k, axis=0).T
        f = np.stack(f, axis=0).T
        self._data = np.stack([A, k, f], axis=-1) # (x, y, wd, AKfs)

        # store ws and wd:
        self.VARS = self.var("VARS")
        self.DATA = self.var("DATA")
        self._wds = np.arange(0., 360., 360 / ns)
        self._wsd = self.ws_bins[1:] - self.ws_bins[:-1]
        self._wss = 0.5 * (self.ws_bins[:-1] + self.ws_bins[1:])
        self._N = len(self._wss) * ns
        data = np.zeros((len(self._wss), ns, 3), dtype=config.dtype_double)
        data[..., 0] = self._wss[:, None]
        data[..., 1] = self._wds[None, :]
        data[..., 2] = self._wsd[:, None]
        data = data.reshape(self._N, 3)
        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = ["ws", "wd", "dws"]
        idata["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data)

        return idata

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

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
        ovars = set([FV.WS, FV.WD])
        ovars.update(self.fixed_vars.keys())
        return list(ovars)
    
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
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        n_pts = n_states * n_targets * n_tpoints
        points = tdata[FC.TARGETS]
        ws = mdata[self.DATA][:, 0]
        wd = mdata[self.DATA][:, 1]
        wsd = mdata[self.DATA][:, 2]

        out = {}

        out[FV.WS] = tdata[FV.WS]
        out[FV.WS][:] = ws[:, None, None]

        out[FV.WD] = tdata[FV.WD]
        out[FV.WD][:] = wd[:, None, None]

        for v, d in self.fixed_vars.items():
            out[v] = tdata[v]
            out[v][:] = d

        # interpolate A, k, f from x, y, wd
        z = points[..., 2].copy()
        points[..., 2] = wd[:, None, None]
        pts = points.reshape(n_pts, 3)
        gvars = (self._x, self._y, self._wds)
        try:
            ipars = dict(bounds_error=True, fill_value=None)
            ipars.update(self.interpn_pars)
            data = interpn(gvars, self._data, pts, **ipars).reshape(
                n_states, n_targets, n_tpoints, 3
            )
        except ValueError as e:
            print(f"\nStates '{self.name}': Interpolation error")
            print("INPUT VARS: (x, y, wd)")
            print(
                "DATA BOUNDS:",
                [float(np.min(d)) for d in gvars],
                [float(np.max(d)) for d in gvars],
            )
            print(
                "EVAL BOUNDS:",
                [float(np.min(p)) for p in pts.T],
                [float(np.max(p)) for p in pts.T],
            )
            print(
                "\nMaybe you want to try the option 'bounds_error=False'? This will extrapolate the data.\n"
            )
            raise e
        
        A = data[..., 0]
        k = data[..., 1]
        f = data[..., 2]
        points[..., 2] = z
        del data, gvars, pts, z, wd

        tdata.add(
            FV.WEIGHT,
            f,
            dims=(FC.STATE, FC.TARGET, FC.TPOINT),
        )
        
        wsA = out[FV.WS] / A
        tdata[FV.WEIGHT] *= wsd[:, None, None] * (
            k / A * wsA ** (k - 1) * np.exp(-wsA ** k)
        )

        return out
    