import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)

from foxes.config import config
from foxes.utils import wd2uv, uv2wd
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
    weight_factor: float, optional
        The factor to multiply the weights with
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
        weight_factor=None,
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
        weight_factor: float, optional
            The factor to multiply the weights with
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
        self.weight_factor = weight_factor
        self.interp_method = interp_method
        self.interp_pars = interp_pars
        self.interp_fallback_nearest = interp_fallback_nearest

        self.variables = [FV.X, FV.Y]
        self.variables += [v for v in self.ovars if v not in self.fixed_vars]
        self.var2ncvar[FV.X] = x_ncvar
        self.var2ncvar[FV.Y] = y_ncvar

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

    def __repr__(self):
        return f"{type(self).__name__}(n_pt={self._n_pt}, n_wd={self._n_wd}, n_ws={self._n_ws})"

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
        points: numpy.ndarray
            The point coordinates, shape (n_points, 2)
            or (n_points, 3) if height is included
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        assert self.x_ncvar in ds.data_vars, \
            f"States '{self.name}': Missing x variable '{self.x_ncvar}' in Dataset, got '{list(ds.data_vars.keys())}'"
        assert self.y_ncvar in ds.data_vars, \
            f"States '{self.name}': Missing y variable '{self.y_ncvar}' in Dataset, got '{list(ds.data_vars.keys())}'"
        assert ds[self.x_ncvar].dims == (self.point_coord,), \
            f"States '{self.name}': x variable '{self.x_ncvar}' has unexpected dimensions {ds[self.x_ncvar].dims}, expected ({self.point_coord},)"
        assert ds[self.y_ncvar].dims == (self.point_coord,), \
            f"States '{self.name}': y variable '{self.y_ncvar}' has unexpected dimensions {ds[self.y_ncvar].dims}, expected ({self.point_coord},)"
        
        points = [
            ds[self.x_ncvar].to_numpy(),
            ds[self.y_ncvar].to_numpy(),
        ]
        if self.h_ncvar is not None:
            assert self.h_ncvar in ds.data_vars, \
                f"States '{self.name}': Missing height variable '{self.h_ncvar}' in Dataset, got '{list(ds.data_vars.keys())}'"
            assert ds[self.h_ncvar].dims == (self.point_coord,), \
                f"States '{self.name}': height variable '{self.h_ncvar}' has unexpected dimensions {ds[self.h_ncvar].dims}, expected ({self.point_coord},)"
            points.append(ds[self.h_ncvar].to_numpy())
        points = np.stack(points, axis=-1)

        cmap = {
            self.states_coord: FC.STATE,
            self.point_coord: FC.POINT,
        }

        data = {}
        for v in variables:
            w = self.var2ncvar.get(v, v)
            if v in data or v in self.fixed_vars or v in [FV.X, FV.Y, FV.H]:
                continue
            if w not in ds.data_vars:
                raise KeyError(f"States '{self.name}': Missing data variable '{w}' in Dataset, got '{list(ds.data_vars.keys())}'")
            
            d = ds[w]
            dims = tuple([cmap[c] for c in d.dims])
            if dims in [(FC.STATE,), (FC.POINT,), (FC.STATE, FC.POINT)]:
                data[v] = (dims, d.to_numpy())
            elif dims == (FC.POINT, FC.STATE):
                data[v] = ((FC.STATE, FC.POINT), np.swapaxes(d.to_numpy(), 0, 1))
            else:
                shps = [(self.states_coord,), (self.point_coord,), (self.states_coord, self.point_coord)]
                raise ValueError(f"States '{self.name}': Failed to map variable '{w}' with dimensions {d.dims} to expected dimensions {shps}")

        if self.weight_factor is not None and FV.WEIGHT in data:
            data[FV.WEIGHT][1][:] *= self.weight_factor

        s = ds[self.states_coord].to_numpy() if self.states_coord in ds else None

        if verbosity > 1 and data is not None:
            print(f"\n{self.name}: Data ranges")
            for v, d in data.items():
                nn = np.sum(np.isnan(d))
                print(
                    f"  {v}: {np.nanmin(d)} --> {np.nanmax(d)}, nans: {nn} ({100 * nn / len(d.flat):.2f}%)"
                )

        return s, points, data

    def _get_data(self, ds, variables, verbosity=0):
        """
        Gets the data from the Dataset and prepares it for calculations.

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
        points: numpy.ndarray
            The point coordinates, shape (n_points, 2)
            or (n_points, 3) if height is included
        data: dict
            The extracted data, keys are dimension tuples,
            values are tuples (DATA key, variables, data_array)     
            where DATA key is the name in the mdata object,
            variables is a list of variable names, and
            data_array is a numpy.ndarray with the data values,
            the last dimension corresponds to the variables
        weights: numpy.ndarray or None
            The weights array, if only state dependent, otherwise
            weights are among data. Shape: (n_states,)

        """
        s, points, data0 = self._read_ds(ds, variables, verbosity)

        weights = None
        if FV.WEIGHT in variables:
            if FV.WEIGHT not in data0:
                raise KeyError(
                    f"States '{self.name}': Missing weights variable '{self.weight_ncvar}' in data, found {sorted(list(data0.keys()))}"
                )
            elif data0[FV.WEIGHT][0] == (self.states_coord,):
                weights = data0.pop(FV.WEIGHT)[1]

        vmap = {
            FC.STATE: FC.STATE,
            FC.POINT: self.var(FC.POINT)
        }

        data = {} # dim: [DATA key, variables, data array]
        for v, (dims, d) in data0.items():
            if dims not in data:
                i = len(data)
                data[dims] = [self.var(f"data{i}"), [], []]
            data[dims][1].append(v)
            data[dims][2].append(d)
        for dims in data.keys():
            data[dims][2] = np.stack(data[dims][2], axis=-1)
        data = {
            tuple([vmap[c] for c in dims] + [self.var(f"vars{i}")]): d
            for i, (dims, d) in enumerate(data.items())
        }

        return s, (points,), data, weights
    
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
            coords=[self.states_coord, self.point_coord],
            variables=self.variables,
            filter_xy=None,
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
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        n_states = fdata.n_states
        n_pts = n_states * n_targets * n_tpoints
        coords = [self.states_coord, self.point_coord]

        # get data for calculation
        (qts,), data, weights = self.get_calc_data(mdata, coords, self.variables)
        n_qts = len(qts)

        # interpolate data to points:
        out = {}
        cmap = {
            FC.STATE: FC.STATE,
            self.var(FC.POINT): FC.POINT,
        }
        for dims, (vrs, d) in data.items():

            idims = tuple([cmap[c] for c in dims[:-1]])
            if idims == (FC.STATE,):
                for i, v in enumerate(vrs):
                    if v in self.ovars:
                        out[v] = np.zeros((n_states, n_targets, n_tpoints), dtype=config.dtype_double)
                        out[v][:] = d[:, None, None, i]
                continue

            elif idims == (FC.POINT,):

                # prepare grid data, add state index to last axis
                gts = qts
                n_vrs = len(vrs)
                n_dms = qts.shape[1]
                n_gts = n_qts

                # prepare evaluation points
                pts = tdata[FC.TARGETS][..., :n_dms].reshape(n_pts, n_dms)

            elif idims == (FC.STATE, FC.POINT):

                # prepare grid data, add state index to last axis
                n_vrs = len(vrs)
                n_dms = qts.shape[1]
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
                ws = (
                    d[..., iws]
                    if FV.WS in vrs
                    else self.fixed_vars[FV.WS]
                )
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
