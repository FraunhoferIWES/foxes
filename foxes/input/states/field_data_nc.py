import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from foxes.core import States
from foxes.tools import wd2uv, uv2wd
import foxes.variables as FV
import foxes.constants as FC

class FieldDataNC(States):

    def __init__(
        self,
        file_pattern,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        states_coord="Time",
        x_coord='UTMX',
        y_coord='UTMY',
        h_coord='height',
        weight_ncvar=None,
        bounds_error=True,
        fill_value=None
    ):
        super().__init__()

        self.file_pattern = file_pattern
        self.states_coord = states_coord
        self.ovars        = output_vars
        self.fixed_vars   = fixed_vars
        self.x_coord      = x_coord
        self.y_coord      = y_coord
        self.h_coord      = h_coord
        self.weight_ncvar = weight_ncvar
        self.bounds_error = bounds_error
        self.fill_value   = fill_value

        self.var2ncvar = {v: var2ncvar.get(v, v) for v in output_vars \
                                if v not in fixed_vars}

        self._inds = None
        self._N    = None
                
    def model_input_data(self, algo):

        if (FV.WS in self.ovars and FV.WD not in self.ovars) \
            or (FV.WS not in self.ovars and FV.WD in self.ovars):
            raise KeyError(f"States '{self.name}': Missing '{FV.WS}' or '{FV.WD}' in output variables {self.ovars}")

        self._weights = None
        with xr.open_mfdataset(self.file_pattern, parallel=True, 
                concat_dim=self.states_coord, combine="nested", 
                data_vars='minimal', coords='minimal', compat='override') as ds:
            
            for c in [self.states_coord, self.x_coord, self.y_coord, self.h_coord]:
                if not c in ds:
                    raise KeyError(f"States '{self.name}': Missing coordinate '{c}' in data")

            self._inds = ds[self.states_coord].values

            for v in self.ovars:
                if v in self.var2ncvar:
                    ncv = self.var2ncvar[v]
                    if not ncv in ds:
                        raise KeyError(f"States '{self.name}': nc variable '{ncv}' not found in data, found: {sorted(list(ds.keys()))}")
                elif v not in self.fixed_vars:
                    raise ValueError(f"States '{self.name}': Variable '{v}' neither found in var2ncvar not in fixed_vars")

            if self.weight_ncvar is not None:
                self._weights = ds[self.weight_ncvar].values
        
        self._N = len(self._inds)

        if self._weights is None:
            self._weights = np.full((self._N, algo.n_turbines), 1./self._N, dtype=FC.DTYPE)

        idata = super().model_input_data(algo)
        idata["coords"][FV.STATE] = self._inds

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

    def index(self):
        """
        The index list
        
        Returns
        -------
        indices : array_like
            The index labels of states, or None for default integers

        """
        return self._inds

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        
        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        return self.ovars

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        
        Returns
        -------
        weights : numpy.ndarray
            The weights, shape: (n_states,)

        """
        return self._weights

    def calculate(self, algo, mdata, fdata, pdata):

        # prepare:
        points   = pdata[FV.POINTS]
        n_pts    = pdata.n_points
        n_states = fdata.n_states
        n_vars   = len(self.var2ncvar)
        if FV.WD in self.fixed_vars:
            n_vars += 1

        # ensure WD and WS get the first two slots of data:
        dkys = {}
        if FV.WS in self.ovars:
            dkys[FV.WD] = 0
        if FV.WS in self.var2ncvar:
            dkys[FV.WS] = 1
        for v in self.var2ncvar:
            if v not in dkys:
                dkys[v] = len(dkys)

        # read data for this chunk:
        i0 = np.where(self._inds==mdata[FV.STATE][0])[0][0]
        s  = slice(i0, i0 + n_states)
        ds = xr.open_mfdataset(self.file_pattern, parallel=False, 
                    concat_dim=self.states_coord, combine="nested", 
                    data_vars='minimal', coords='minimal', compat='override'
                ).isel({self.states_coord: s})
        
        # prepare data:
        x      = ds[self.x_coord].values
        y      = ds[self.y_coord].values
        h      = ds[self.h_coord].values
        n_x    = len(x)
        n_y    = len(y)
        n_h    = len(h)
        data   = np.zeros((n_states, n_h, n_y, n_x, n_vars), dtype=FC.DTYPE)
        cor_xy = (self.states_coord, self.h_coord, self.x_coord, self.y_coord) 
        cor_yx = (self.states_coord, self.h_coord, self.y_coord, self.x_coord)
        for v, ncv in self.var2ncvar.items():
            if ds[ncv].dims == cor_yx:
                data[..., dkys[v]] = ds[ncv][:]
            elif ds[ncv].dims == cor_xy:
                data[..., dkys[v]] = np.swapaxes(ds[ncv].values, 2, 3)
            else:
                raise ValueError(f"States '{self.name}': Wrong coordinate order for variable '{ncv}': Found {ds[ncv].dims}, expecting {cor_xy} or {cor_yx}")
        if FV.WD in self.fixed_vars:
            data[..., dkys[FV.WD]] = np.full((n_states, n_h, n_y, n_x), self.fixed_vars[FV.WD], dtype=FC.DTYPE)
        del ds
        
        # translate WS, WD into U, V:
        if FV.WD in self.ovars and FV.WS in self.ovars:
            wd = data[..., dkys[FV.WD]]
            ws = data[..., dkys[FV.WS]] if FV.WS in dkys else self.fixed_vars[FV.WS]
            data[..., :2] = wd2uv(wd, ws, axis=-1)
            del ws, wd

        # prepare points:
        sts = np.arange(n_states)
        pts = np.append(points, np.zeros((n_states, n_pts, 1), dtype=FC.DTYPE), axis=2)
        pts[:, :, 3] = sts[:, None]
        pts = pts.reshape(n_states*n_pts, 4)
        pts = np.flip(pts, axis=1)

        # interpolate:
        gvars = (sts, h, y, x)
        iterp = RegularGridInterpolator(gvars, data, 
                    bounds_error=self.bounds_error, fill_value=self.fill_value)
        try:
            data = iterp(pts).reshape(n_states, n_pts, n_vars)
        except ValueError as e:
            print(f"\n\nStates '{self.name}': Interpolation error")
            print("INPUT VARS : (state, heights, y, x)")
            print("DATA BOUNDS:", [np.min(d) for d in gvars], [np.max(d) for d in gvars])
            print("EVAL BOUNDS:", [np.min(p) for p in pts.T], [np.max(p) for p in pts.T])
            raise e
        del pts, iterp, x, y, h, gvars

        # set output:
        out = {}
        if FV.WD in self.ovars and FV.WS in self.ovars:
            uv = data[..., :2]
            out[FV.WS] = np.linalg.norm(uv, axis=-1)
            out[FV.WD] = uv2wd(uv, axis=-1)
            del uv
        for v in self.ovars:
            if v not in out:
                if v in dkys:
                    out[v] = data[..., dkys[v]]
                else:
                    out[v] = np.full((n_states, n_pts), self.fixed_vars[v], dtype=FC.DTYPE)

        return out
