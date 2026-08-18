import numpy as np
import pandas as pd
import xarray as xr
import threading
from copy import copy
from collections.abc import Callable, Generator
from scipy.interpolate import interpn
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

from foxes.core import (
    Algorithm,
    FData,
    LoadedData,
    MData,
    States,
    TData,
    map_with_engine,
)
from foxes.utils import import_module
from foxes.data import STATES, StaticData
from foxes.utils.wind_dir import uv2wd, wd2uv
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC


# Serialize netcdf4 file opens to avoid HDF5 attribute access errors in threaded reads.
_NETCDF4_OPEN_LOCK = threading.Lock()


def _read_nc_file(
    fpath: Path,
    coords: list[str],
    vars: list[str] | None,
    nc_engine: str | None,
    sel: dict[str, object] | None,
    isel: dict[str, object] | None,
    mode: str,
    drop_vars: list[str] | None = None,
    sort: bool | list[str] | None = False,
    check_input_nans: bool = True,
    preprocess: Callable[[xr.Dataset], xr.Dataset] | None = None,
) -> xr.Dataset | np.ndarray | None:
    """Helper function for nc file reading"""
    open_lock = _NETCDF4_OPEN_LOCK if nc_engine == "netcdf4" else nullcontext()
    result: xr.Dataset | np.ndarray | None
    with open_lock:
        with xr.open_dataset(fpath, drop_variables=drop_vars, engine=nc_engine) as data:
            # Ensure deterministic ascending coordinate order for all dimensions.
            if sort is not None:
                if isinstance(sort, bool):
                    if sort:
                        for d in data.dims:
                            if d in data.coords:
                                data = data.sortby(d)
                elif isinstance(sort, list):
                    for d in sort:
                        assert d in data.dims, (
                            f"Cannot sort by dimension '{d}' in file {fpath}, not found among dimensions {list(data.dims)}"
                        )
                        data = data.sortby(d)
                else:
                    raise ValueError(
                        f"Invalid sort argument of type {type(sort).__name__}, expected bool or list of str"
                    )

            for c in coords:
                if c is not None and c not in data.sizes:
                    raise KeyError(
                        f"Missing coordinate '{c}' in file {fpath}, got: {list(data.sizes.keys())}"
                    )

            if preprocess is not None:
                data = preprocess(data)

            if mode == "minimal":
                c = coords[0]
                try:
                    if isel is not None and c in isel:
                        data = data.isel(indexers={c: isel[c]})
                    if sel is not None and c in sel:
                        data = data.sel(indexers={c: sel[c]})
                    result = data[c].to_numpy()
                except KeyError:
                    result = None

            else:
                if vars is not None:
                    data = data[vars]
                data.attrs = {}

                try:
                    if isel is not None and len(isel):
                        isel = {c: s for c, s in isel.items() if c in data.sizes}
                        data = data.isel(indexers=isel)
                    if sel is not None and len(sel):
                        sel = {c: s for c, s in sel.items() if c in data.sizes}
                        data = data.sel(indexers=sel)
                except KeyError:
                    return None

                if min(data.sizes.values()) == 0:
                    result = None
                elif mode == "load":
                    result = data.load()
                elif mode == "lazy":
                    result = data
                else:
                    raise NotImplementedError(
                        f"Mode '{mode}' not implemented, choices: minimal, lazy, load"
                    )

    if result is not None and mode != "minimal" and check_input_nans:
        assert isinstance(result, xr.Dataset)
        for v_raw in result.data_vars:
            v = str(v_raw)
            data_array = cast(xr.DataArray, result[v])
            nan_mask = np.isnan(data_array.to_numpy())
            if nan_mask.any():
                i = tuple([j[0] for j in np.where(nan_mask)])
                print("\n\nError: NaN data found in input data:")
                print(f"  File: {fpath}\n")
                print(f"  Variable: {v}")
                for ic, c_raw in enumerate(data_array.dims):
                    c = str(c_raw)
                    print(f"  {c}: {result[c].to_numpy()[i[ic]]}")
                print("\n\n")
                raise ValueError(
                    f"States: NaN data found in input data for variable '{v}' with dims {data_array.dims} in file {fpath} at index {i}"
                )

    return result


class DatasetStates(States):
    """
    Abstract base class for heterogeneous ambient states that
    are based on data from NetCDF files or an xarray Dataset.

    Attributes
    ----------
    data_source
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    ovars
        The output variables
    var2ncvar
        Mapping from variable names to variable names
        in the nc file
    fixed_vars
        Uniform values for output variables, instead
        of reading from data
    time_format
        The datetime parsing format string
    bounds_extra_space
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    height_bounds
        The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
    sel
        Subset selection via xr.Dataset.sel()
    isel
        Subset selection via xr.Dataset.isel()
    weight_factor
        The factor to multiply the weights with
    sort
        Whether to sort the data by the state coordinate, or selected coordinates
    check_times
        Whether to check the time coordinates for consistency
    check_input_nans
        Whether to check input data for NaNs
    preprocess_nc
        A function to preprocess the netcdf Dataset before use
    force_keep_vars
        Variables to remove from the drop_vars list when reading the nc files
    interp_pars
        Additional parameters the interpolation


    """

    def __init__(
        self,
        data_source: str | Path | xr.Dataset,
        output_vars: list[str],
        var2ncvar: dict[str, str] = {},
        fixed_vars: dict[str, float] = {},
        load_mode: str = "preload",
        time_format: str | None = None,
        bounds_extra_space: float | str | None = 100.0,
        height_bounds: tuple[float, float] | None = None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        weight_factor: float | None = None,
        check_times: bool = True,
        check_input_nans: bool = True,
        sort: bool | list[str] = False,
        preprocess_nc: Callable[[xr.Dataset], xr.Dataset] | None = None,
        force_keep_vars: list[str] | None = None,
        interp_pars: dict[str, bool | float | str | None] = {},
        **kwargs: object,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The data or the file search pattern, should end with
            suffix '.nc'. One or many files.
        output_vars
            The output variables
        var2ncvar
            Mapping from variable names to variable names
            in the nc file
        fixed_vars
            Uniform values for output variables, instead
            of reading from data
        load_mode
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculation
        time_format
            The datetime parsing format string
        bounds_extra_space
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'. If None,
            all points from the input data are used
        height_bounds
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        sel
            Subset selection via xr.Dataset.sel()
        isel
            Subset selection via xr.Dataset.isel()
        weight_factor
            The factor to multiply the weights with
        check_times
            Whether to check the time coordinates for consistency
        check_input_nans
            Whether to check input data for NaNs, otherwise NaNs are removed
        sort
            Whether to sort the data by the state coordinate, or selected coordinates
        preprocess_nc
            A function to preprocess the netcdf Dataset before use
        force_keep_vars
            Variables to remove from the drop_vars list when reading the nc files
        interp_pars
            Additional parameters the interpolation
        kwargs
            Additional arguments for the base class

        """
        super().__init__(load_mode=load_mode, **kwargs)

        self.ovars = list(output_vars)
        self.fixed_vars = fixed_vars
        self.var2ncvar = var2ncvar
        self.time_format = time_format
        self.sel = sel
        self.isel = isel
        self.weight_factor = weight_factor
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds
        self.sort = sort
        self.check_times = check_times
        self.check_input_nans = check_input_nans
        self.preprocess_nc = preprocess_nc
        self.interp_pars = interp_pars if interp_pars is not None else {}
        self.variables = [v for v in self.ovars if v not in self.fixed_vars]
        self.force_keep_vars = force_keep_vars if force_keep_vars is not None else []

        # keep U and V in variables, but replace by WS, WD in output variables:
        if FV.U in self.ovars or FV.V in self.ovars:
            assert FV.U in self.ovars and FV.V in self.ovars, (
                f"States '{self.name}': Require both {FV.U} and {FV.V} in output_vars to compute wind direction, got {self.ovars}"
            )
            assert FV.WS not in self.ovars and FV.WD not in self.ovars, (
                f"States '{self.name}': Cannot have {FV.U} and {FV.V} together with {FV.WS} or {FV.WD} in output_vars, got {self.ovars}"
            )
            self.ovars[self.ovars.index(FV.U)] = FV.WS
            self.ovars[self.ovars.index(FV.V)] = FV.WD

        self._N: int | None = None
        self._inds: np.ndarray | None = None
        self._cmap: dict[str, str] = {}
        self._files_maxi: dict[Path, int] = {}
        self._input_sizes: list[int] = []
        self.__data_source = data_source

    @property
    def data_source(self) -> str | Path | xr.Dataset:
        """
        The data source

        Returns
        -------
        data_source
            The data source

        """
        if self.load_mode in ["preload", "fly"] and self.running:
            raise ValueError(
                f"States '{self.name}': Cannot access data_source while running for load mode '{self.load_mode}'"
            )
        return self.__data_source

    def _read_ds(
        self,
        ds: xr.Dataset,
        cmap: dict[str, str] | None = None,
        verbosity: int = 0,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, tuple[tuple[str, ...], np.ndarray]],
    ]:
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds
            The Dataset to read data from
        cmap
            A mapping from foxes variable names to Dataset dimension names, if None, use self._cmap
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        coords
            keys: Foxes variable names, values: 1D coordinate value arrays
        data
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where each value contains dimensions and data values

        """
        cmap = cmap if cmap is not None else self._cmap
        data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for v, w in self._vars.items():
            if w in ds.data_vars:
                d = ds[w]
                i = [d.dims.index(c) for c in cmap.values() if c in d.dims]
                assert len(i) == len(d.dims), (
                    f"States '{self.name}': Variable '{w}' has dimensions {d.dims}, but not all of them are in the coordinate mapping {cmap}"
                )
                dms = tuple([v for v, c in cmap.items() if c in d.dims])
                j = list(range(len(i)))
                if i == j:
                    data[v] = (dms, d.to_numpy())
                elif len(i) == 2:
                    data[v] = (dms, np.swapaxes(d.to_numpy(), 0, 1))
                else:
                    data[v] = (dms, np.moveaxis(d.to_numpy(), i, j))
            else:
                raise KeyError(
                    f"States '{self.name}': Variable '{w}' not found in data, available variables: {list(ds.data_vars)}"
                )
        coords: dict[str, np.ndarray] = {
            v: ds[c].to_numpy() for v, c in cmap.items() if c in ds.coords
        }

        if FC.STATE in coords and self.time_format is not None:
            coords[FC.STATE] = pd.to_datetime(
                coords[FC.STATE], format=self.time_format
            ).to_numpy()

        if verbosity > 1:
            if len(coords):
                print(f"\n{self.name}: Coordinate ranges")
                for c, coord_data in coords.items():
                    print(f"  {c}: {np.min(coord_data)} --> {np.max(coord_data)}")
            print(f"\n{self.name}: Data ranges")
            for v, data_entry in data.items():
                nn = np.sum(np.isnan(data_entry[1]))
                print(
                    f"  {v}: {np.nanmin(data_entry[1])} --> {np.nanmax(data_entry[1])}, nans: {nn} ({100 * nn / len(data_entry[1].flat):.2f}%)"
                )

        return coords, data

    def _get_data(
        self,
        ds: xr.Dataset,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        verbosity: int = 0,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[tuple[str, ...], tuple[str, list[str], np.ndarray]],
        np.ndarray | None,
    ]:
        """
        Gets the data from the Dataset and prepares it for calculations.

        Parameters
        ----------
        ds
            The Dataset to read data from
        bounds_extra_space
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        coords
            keys: Foxes variable names, values: 1D coordinate value arrays
        data
            The extracted data, keys are dimension tuples,
            values are tuples (DATA key, variables, data_array)
            where DATA key is the name in the mdata object,
            variables are the variable names, and
            data_array contains the data values,
            the last dimension corresponds to the variables
        weights
            The weights array, if only state dependent, otherwise
            weights are among data. Shape: (n_states,)

        """
        coords, data0 = self._read_ds(ds, verbosity=verbosity)

        weights = None
        if FV.WEIGHT in self._vars:
            assert FV.WEIGHT in data0, (
                f"States '{self.name}': Missing weights variable '{FV.WEIGHT}' in data, found {sorted(list(data0.keys()))}"
            )
            if self.weight_factor is not None:
                dims, values = data0[FV.WEIGHT]
                data0[FV.WEIGHT] = (dims, values * self.weight_factor)
            if data0[FV.WEIGHT][0] == (FC.STATE,):
                weights = data0.pop(FV.WEIGHT)[1]

        data_groups: dict[tuple[str, ...], tuple[str, list[str], list[np.ndarray]]] = {}
        for v, (dims, d) in data0.items():
            if dims not in data_groups:
                i = len(data_groups)
                data_groups[dims] = (self.var(f"data{i}"), [], [])
            data_groups[dims][1].append(v)
            data_groups[dims][2].append(d)
        data: dict[tuple[str, ...], tuple[str, list[str], np.ndarray]] = {
            tuple(list(dims) + [f"vars{i}"]): (
                data_key,
                variables,
                np.stack(arrays, axis=-1),
            )
            for i, (dims, (data_key, variables, arrays)) in enumerate(
                data_groups.items()
            )
        }

        return coords, data, weights

    def _find_xy_bounds(
        self, algo: Algorithm, bounds_extra_space: float | str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper function to determine x/y bounds with extra space."""
        return algo.farm.get_xy_bounds(extra_space=bounds_extra_space, algo=algo)

    def _update_loaded_state_indices(self, loaded_data: LoadedData | None) -> None:
        """Store only non-default state indices in loaded data."""
        if loaded_data is None or self._inds is None:
            return

        inds = np.asarray(self._inds)
        is_default = np.issubdtype(inds.dtype, np.number) and np.array_equal(
            inds, np.arange(self._N, dtype=inds.dtype)
        )
        if is_default:
            loaded_data["coords"].pop(FC.STATE, None)
        else:
            loaded_data["coords"][FC.STATE] = self._inds

    def preproc_first(
        self,
        algo: Algorithm,
        data: xr.Dataset,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        loaded_data: LoadedData | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Preprocesses the first file.

        Parameters
        ----------
        algo
            The calculation algorithm
        data
            The dataset to preprocess
        bounds_extra_space
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
            loaded_data
            If given, optionally add to this loaded data dict with entries
            {"coords": {}, "data_vars": {}, "extra_data": {}}
        verbosity
            The verbosity level, 0 = silent

        """

        # check for UTM zone:
        if "utm_number" in data or "utm_letter" in data:
            assert "utm_number" in data and "utm_letter" in data, (
                f"States '{self.name}': Require both 'utm_number' and 'utm_letter' in data to set UTM zone, found {list(data.keys())}"
            )
            config.set_utm_zone(
                int(data["utm_number"].values), str(data["utm_letter"].values)
            )

        # check if needed:
        if bounds_extra_space == np.inf and height_bounds == np.inf:
            return

        # find vertical bounds:
        if FV.H in self._cmap:
            if height_bounds is None:
                H = algo.farm.get_hub_heights(algo)
                D = algo.farm.get_rotor_diameters(algo)
                H = np.stack((H - 0.5 * D, H + 0.5 * D), axis=-1)
                height_bounds = (np.min(H), np.max(H))
                del H, D
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting heights to {height_bounds[0]} - {height_bounds[1]} m"
                )
            self._heights = data[self._cmap[FV.H]].to_numpy()
            if (
                np.min(self._heights) > height_bounds[0]
                or np.max(self._heights) < height_bounds[1]
            ):
                raise ValueError(
                    f"States '{self.name}': Height bounds {height_bounds} m are outside of data height range {np.min(self._heights)} - {np.max(self._heights)} m"
                )
            ch = self._cmap[FV.H]
            if self.isel is None or ch not in self.isel:
                i0 = 0
                while (
                    i0 < len(self._heights) - 1
                    and self._heights[i0 + 1] <= height_bounds[0]
                ):
                    i0 += 1
                i1 = len(self._heights) - 1
                while i1 > 0 and self._heights[i1 - 1] >= height_bounds[1]:
                    i1 -= 1
                if i0 == i1:
                    i0 = max(0, i0 - 1)
                    i1 = min(len(self._heights) - 1, i1 + 1)
                if self.isel is None:
                    self.isel = {}
                self.isel.update({ch: slice(i0, i1 + 1)})
            self._heights = data[ch].isel({ch: self.isel[ch]}).to_numpy()
            if verbosity > 0:
                print(
                    f"States '{self.name}': Selected {ch} = {self._heights} ({len(self._heights)} heights)"
                )

        # find horizontal bounds:
        if bounds_extra_space is not None:
            assert FV.X in self._cmap, (
                f"States '{self.name}': x coordinate '{FV.X}' not in cmap {self._cmap}"
            )
            assert FV.Y in self._cmap, (
                f"States '{self.name}': y coordinate '{FV.Y}' not in cmap {self._cmap}"
            )
            xy_min, xy_max = self._find_xy_bounds(algo, bounds_extra_space)
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting xy to bounds {xy_min} - {xy_max}"
                )
            for v, i in zip((FV.X, FV.Y), (0, 1)):
                if self.isel is None or self._cmap[v] not in self.isel:
                    x0, x1 = xy_min[i], xy_max[i]
                    x = data[self._cmap[v]].to_numpy()
                    i0 = 0
                    while i0 < len(x) - 1 and x[i0 + 1] <= x0:
                        i0 += 1
                    i1 = len(x) - 1
                    while i1 > 0 and x[i1 - 1] >= x1:
                        i1 -= 1
                    if i0 == i1:
                        i0 = max(0, i0 - 1)
                        i1 = min(len(x) - 1, i1 + 1)
                    if self.isel is None:
                        self.isel = {}
                    self.isel.update({self._cmap[v]: slice(i0, i1 + 1)})
                if verbosity > 0:
                    hv = (
                        data[self._cmap[v]]
                        .isel({self._cmap[v]: self.isel[self._cmap[v]]})
                        .to_numpy()
                    )
                    print(
                        f"States '{self.name}': Selected {self._cmap[v]} = {hv[0]} ... {hv[-1]} ({len(hv)} points)"
                    )

        # optionally add coordinates to loaded data:
        if loaded_data is not None:
            for c in (FV.X, FV.Y, FV.H):
                if c in self._cmap:
                    cc = self._cmap[c]
                    d = (
                        data[cc].isel({cc: self.isel[cc]})
                        if self.isel is not None and cc in self.isel
                        else data[cc]
                    )
                    loaded_data["coords"][cc] = d.to_numpy()
                    del d

    def __load_files(
        self,
        algo: Algorithm,
        bounds_extra_space: float | str | None,
        height_bounds: tuple[float, float] | None,
        loaded_data: LoadedData | None = None,
        verbosity: int = 0,
    ) -> xr.Dataset:
        """Initial loading of all files, as needed by load mode"""

        assert FC.STATE in self._cmap, (
            f"States '{self.name}': States coordinate '{FC.STATE}' not in cmap {self._cmap}"
        )
        states_coord = self._cmap[FC.STATE]

        def _update_vars(ds: xr.Dataset, vars: dict[str, str]) -> dict[str, str]:
            """Helper function to automatically update variables"""
            # automatically switch TI to TKE, if TI not provided:
            if FV.TI in vars:
                cti = self.var2ncvar.get(FV.TI, FV.TI)
                ctke = self.var2ncvar.get(FV.TKE, FV.TKE)
                if cti not in ds.data_vars and ctke in ds.data_vars:
                    if verbosity > 1:
                        print(
                            f"States '{self.name}': Variable '{cti}' not found, but '{ctke}' found, using it as {FV.TKE} for the calculation of {FV.TI}"
                        )
                    vars[FV.TKE] = ctke
                    del vars[FV.TI]
            return vars

        if not isinstance(self.data_source, xr.Dataset):
            # check static data:
            fpath = get_input_path(self.data_source)
            if "*" not in str(self.data_source):
                if not fpath.is_file():
                    static_path = StaticData().get_file_path(
                        STATES, fpath.name, check_raw=False
                    )
                    assert static_path is not None
                    fpath = static_path

            # find files:
            prt = fpath.resolve().parent
            glb = fpath.name
            while "*" in str(prt):
                glb = prt.name + "/" + glb
                prt = prt.parent
            files = sorted(list(prt.glob(glb)))
            coords = list(self._cmap.values())
            vars = {v: self.var2ncvar.get(v, v) for v in self.variables}

            # pre-process first file:
            data_first = None
            file_i = 0
            while data_first is None and file_i < len(files):
                fpath = files[file_i]
                data_first = _read_nc_file(
                    fpath,
                    coords=coords,
                    vars=None,
                    nc_engine=config.nc_engine,
                    isel=self.isel,
                    sel=self.sel,
                    mode="load",
                    drop_vars=None,
                    sort=self.sort,
                    check_input_nans=False,
                    preprocess=None,
                )
                file_i += 1
            assert data_first is not None, (
                f"States '{self.name}': No valid data sources found."
            )
            assert isinstance(data_first, xr.Dataset)
            if verbosity > 0:
                print(f"States '{self.name}': Preprocessing file", fpath.name)
            vars = _update_vars(data_first, vars)
            self._vars = vars
            self.drop_vars = [
                v
                for v in data_first.data_vars
                if v not in coords + list(vars.values())
                and v not in self.force_keep_vars
            ]
            if len(self.drop_vars) > 0 and verbosity > 0:
                print(f"States '{self.name}': Keeping variables  {list(vars.values())}")
                print(f"States '{self.name}': Dropping variables {self.drop_vars}")
            if self.preprocess_nc is not None:
                data_first = self.preprocess_nc(data_first)
            self.preproc_first(
                algo,
                data=data_first,
                bounds_extra_space=bounds_extra_space,
                height_bounds=height_bounds,
                loaded_data=loaded_data,
                verbosity=verbosity,
            )
            del data_first

            # read files:
            if verbosity > 0:
                if self.load_mode == "preload":
                    print(
                        f"States '{self.name}': Reading data from '{self.data_source}'"
                    )
                elif self.load_mode == "lazy":
                    print(
                        f"States '{self.name}': Reading header from '{self.data_source}'"
                    )
                else:
                    print(
                        f"States '{self.name}': Reading states from '{self.data_source}'"
                    )

            mode = {"fly": "minimal", "lazy": "lazy", "preload": "load"}[self.load_mode]
            data = map_with_engine(
                _read_nc_file,
                files,
                coords=coords,
                vars=list(vars.values()),
                nc_engine=config.nc_engine,
                isel=self.isel,
                sel=self.sel,
                mode=mode,
                drop_vars=self.drop_vars,
                sort=self.sort,
                check_input_nans=self.check_input_nans,
                preprocess=self.preprocess_nc,
            )

            def _len_ds(ds: xr.Dataset | np.ndarray) -> int:
                """Helper function to get the number of states"""
                return ds.sizes[states_coord] if isinstance(ds, xr.Dataset) else len(ds)

            file_data = [
                (f, ds)
                for f, ds in zip(files, data)
                if ds is not None and _len_ds(ds) > 0
            ]
            files = [f for f, _ in file_data]
            data = [ds for _, ds in file_data]
            assert len(data) > 0, f"States '{self.name}': No valid data sources found."

            if self.load_mode in ["preload", "lazy"]:
                if self.load_mode == "lazy":
                    try:
                        data = [ds.chunk() for ds in data]
                    except (ModuleNotFoundError, ValueError) as e:
                        import_module("dask")
                        raise e
                if len(data) == 1:
                    data = data[0]
                else:
                    data = xr.concat(
                        data,
                        dim=states_coord,
                        coords="minimal",
                        data_vars="minimal",
                        compat="equals",
                        join="exact",
                        combine_attrs="drop",
                    )
                if self._inds is None or len(self._inds) != data.sizes[states_coord]:
                    self._inds = (
                        np.arange(self._N)
                        if self._N is not None
                        else data[states_coord].to_numpy()
                    )
                else:
                    self._inds = data[states_coord].to_numpy()
                self._N = len(self._inds)

            elif self.load_mode == "fly":
                file_inds = [cast(np.ndarray, inds) for inds in data]
                self._files_maxi = {f: len(inds) for f, inds in zip(files, file_inds)}
                self._input_sizes = list(self._files_maxi.values())
                self._inds = np.concatenate(file_inds, axis=0)
                self._N = len(self._inds)

            else:
                raise KeyError(
                    f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
                )

            if self.time_format is not None:
                self._inds = pd.to_datetime(
                    self._inds, format=self.time_format
                ).to_numpy()

        # given data is already Dataset:
        else:
            data = self.data_source
            if self.preprocess_nc is not None:
                data = self.preprocess_nc(data)
            if self.isel is not None and len(self.isel):
                hisel = {c: s for c, s in self.isel.items() if c in data.sizes}
                if len(hisel):
                    data = data.isel(indexers=hisel)
            if self.sel is not None and len(self.sel):
                hsel = {c: s for c, s in self.sel.items() if c in data.sizes}
                if len(hsel):
                    data = data.sel(indexers=hsel)
            self.preproc_first(
                algo,
                data=data,
                bounds_extra_space=bounds_extra_space,
                height_bounds=height_bounds,
                loaded_data=loaded_data,
                verbosity=verbosity,
            )
            if self._inds is None or len(self._inds) != data.sizes[states_coord]:
                self._inds = (
                    np.arange(self._N)
                    if self._N is not None
                    else data[states_coord].to_numpy()
                )
            else:
                self._inds = data[states_coord].to_numpy()
            self._N = len(self._inds)
            self._vars = {v: self.var2ncvar.get(v, v) for v in self.variables}
            self._vars = _update_vars(data, self._vars)

        # make sure state indices are sorted ascending:
        def _is_sorted(a: np.ndarray) -> bool:
            return bool(np.all(a[:-1] <= a[1:]))

        if self.check_times and not _is_sorted(self._inds):
            print("\n\nError with state indices, not sorted:\n")
            print(f"State {0:07d}: {self._inds[0]}")
            for i in range(1, self._N):
                print(f"State {i:07d}: {self._inds[i]}")
                if self._inds[i] < self._inds[i - 1]:
                    break
            print()
            raise ValueError(
                f"States '{self.name}': State indices are not sorted ascending: {self._inds[i - 1]} > {self._inds[i]} at position {i - 1}"
            )

        self._update_loaded_state_indices(loaded_data)

        return data

    def gen_states_split_size(self) -> Generator[int | None, None, None]:
        """
        Generator for suggested states split sizes for output writing.

        Yields
        ------
        split_size
            The suggested split size, or None for no splitting

        """
        for size in self._input_sizes:
            yield size

    def load_data(  # type: ignore[override]
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all data required for model calculations.

        The function adds to loaded_data.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            It contains coordinate data, model variables, and additional data.
        force
            Overwrite existing data
        bounds_extra_space
            The extra space in meters to add to the horizontal wind farm bounds
        height_bounds
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        verbosity
            The verbosity level, 0 = silent

        """

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        # check if already loaded:
        self.META = self.var("meta")
        edata = loaded_data["extra_data"]
        if not force and self.META in edata:
            return
        edata[self.META] = {}

        # load data from files or dataset:
        bounds_extra_space = (
            bounds_extra_space
            if bounds_extra_space is not None
            else self.bounds_extra_space
        )
        height_bounds = (
            height_bounds if height_bounds is not None else self.height_bounds
        )
        loaded_dataset = self.__load_files(
            algo,
            bounds_extra_space=bounds_extra_space,
            height_bounds=height_bounds,
            loaded_data=loaded_data,
            verbosity=verbosity,
        )

        # store data for preload mode:
        if self.load_mode == "preload":
            coords, prepared_data, w = self._get_data(
                loaded_dataset,
                bounds_extra_space=bounds_extra_space,
                height_bounds=height_bounds,
                verbosity=verbosity,
            )

            self._update_loaded_state_indices(loaded_data)

            vmap = {FC.STATE: FC.STATE, FC.TURBINE: FC.TURBINE}
            for c, d in coords.items():
                c = self.var(c) if c not in [FC.STATE, FC.TURBINE] else c
                if isinstance(d, tuple):
                    dims, values = d
                    dms = tuple([vmap.get(v, self.var(v)) for v in dims])
                    loaded_data["coords"][c] = (dms, values)
                else:
                    loaded_data["coords"][c] = d
            if w is not None:
                loaded_data["data_vars"][FV.WEIGHT] = ((FC.STATE,), w)

            edata[self.META]["data_keys"] = []
            for dims, prepared_entry in prepared_data.items():
                dms = tuple([vmap.get(c, self.var(c)) for c in dims])
                loaded_data["coords"][dms[-1]] = np.asarray(
                    prepared_entry[1], dtype=str
                )
                loaded_data["data_vars"][prepared_entry[0]] = (
                    dms,
                    prepared_entry[2],
                )
                edata[self.META]["data_keys"].append(prepared_entry[0])

        # store data for lazy mode:
        elif self.load_mode == "lazy":
            self.__lazy_data = loaded_dataset

    def load_chunk_data(  # type: ignore[override]
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> None:
        """
        Load chunk data according to load mode.

        This function adds data to mdata.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data

        """

        # load sub model chunk data:
        super().load_chunk_data(algo, mdata, fdata, tdata)

        # prepare:
        assert FC.STATE in self._cmap, (
            f"States '{self.name}': States coordinate '{FC.STATE}' not in cmap {self._cmap}"
        )
        states_coord = self._cmap[FC.STATE]
        n_states = mdata.n_states
        assert n_states is not None
        edata = mdata.extra_data

        # preloading already done:
        if self.load_mode == "preload":
            return

        # lazy loading:
        elif self.load_mode == "lazy":
            i0 = mdata.states_i0(counter=True)
            assert i0 is not None
            s = slice(i0, i0 + n_states)
            data = self.__lazy_data.isel({states_coord: s}).load()

        # loading this chunk's data on the fly:
        elif self.load_mode == "fly":
            chunk_data: list[xr.Dataset] = []
            i0 = mdata.states_i0(counter=True)
            assert i0 is not None
            i1 = i0 + n_states
            j0 = 0
            for fpath, n in self._files_maxi.items():
                if i0 < j0 or i0 == i1:
                    break
                else:
                    j1 = j0 + n
                    if i0 < j1:
                        a = i0 - j0
                        b = min(i1, j1) - j0
                        assert b > a, (
                            f"States '{self.name}': Invalid state indices for file {fpath}: (i0, i1, j0, j1, a, b) = {(i0, i1, j0, j1, a, b)}"
                        )
                        isel = copy(self.isel) if self.isel is not None else {}
                        isel[states_coord] = slice(a, b)

                        d = _read_nc_file(
                            fpath,
                            coords=list(self._cmap.values()),
                            vars=list(self._vars.values()),
                            nc_engine=config.nc_engine,
                            isel=isel,
                            sel=self.sel,
                            mode="load",
                            drop_vars=[str(v) for v in self.drop_vars],
                            sort=self.sort,
                            check_input_nans=self.check_input_nans,
                            preprocess=self.preprocess_nc,
                        )
                        if d is not None:
                            assert isinstance(d, xr.Dataset)
                            chunk_data.append(d)
                        else:
                            raise ValueError(
                                f"States '{self.name}': Failed to read data for file {fpath}"
                            )
                        del d
                        i0 += b - a
                    j0 = j1

            assert i0 == i1, (
                f"States '{self.name}': Missing states for load_mode '{self.load_mode}': (i0, i1) = {(i0, i1)}"
            )
            assert len(chunk_data) > 0, (
                f"States '{self.name}': No data read for load_mode '{self.load_mode}'"
            )
            if len(chunk_data) == 1:
                data = chunk_data[0]
            else:
                data = xr.concat(
                    chunk_data,
                    dim=states_coord,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="exact",
                    combine_attrs="drop",
                )

        else:
            raise NotImplementedError(
                f"States '{self.name}': load mode '{self.load_mode}' not implemented"
            )

        # add data to mdata:
        if self.load_mode != "preload":
            coords, prepared_data, weights = self._get_data(data, verbosity=0)
            vmap = {FC.STATE: FC.STATE, FC.TURBINE: FC.TURBINE}
            edata[self.META]["data_keys"] = []
            for dims, prepared_entry in prepared_data.items():
                dms = tuple([vmap.get(c, self.var(c)) for c in dims])

                mdata[dms[-1]] = np.asarray(prepared_entry[1], dtype=str)
                mdata.dims[dms[-1]] = (dms[-1],)

                mdata[prepared_entry[0]] = prepared_entry[2]
                mdata.dims[prepared_entry[0]] = dms

                edata[self.META]["data_keys"].append(prepared_entry[0])

            if weights is not None:
                mdata[FV.WEIGHT] = weights
                mdata.dims[FV.WEIGHT] = (FC.STATE,)

            for c, d in coords.items():
                c = self.var(c) if c not in [FC.STATE, FC.TURBINE] else c
                mdata[c] = d
                mdata.dims[c] = (c,)

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        data_stash
            Large data stash, this function adds data here, if given.
            Key: model name. Value: dict, large model data
        sel
            The subset selection dictionary
        isel
            The index subset selection dictionary
        verbosity
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data_stash[self.name] = dict(
                inds=self._inds,
            )
            if self.load_mode == "preload":
                data_stash[self.name]["data_source"] = self.__data_source
                del self.__data_source
        del self._inds

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo
            The calculation algorithm
        data_stash
            Reconstruct model data from this stash, if given.
            Key: model name. Value: dict, large model data
        sel
            The subset selection dictionary
        isel
            The index subset selection dictionary
        verbosity
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data = data_stash[self.name]
            self._inds = cast(np.ndarray | None, data.pop("inds"))

            if self.load_mode == "preload":
                self.__data_source = cast(
                    str | Path | xr.Dataset, data.pop("data_source")
                )

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        return self.ovars

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int
            The total number of states

        """
        assert self._N is not None
        return self._N

    def index(self) -> np.ndarray | None:
        """
        The index list

        Returns
        -------
        indices
            The index labels of states, or None for default integers

        """
        if self.running:
            raise ValueError(f"States '{self.name}': Cannot access index while running")
        return self._inds

    def get_grid_points(
        self,
        loaded_data: LoadedData | None = None,
        mdata: MData | None = None,
        all_heights: bool = True,
        height: float | None = None,
    ) -> np.ndarray:
        """
        Returns the grid points from the mdata object.

        Parameters
        ----------
        loaded_data
            The loaded data dictionary
        mdata
            The model data
        all_heights
            If True, return all heights, otherwise only the highest.
        height
            The height to use. If None, the highest height is used if
            all_heights is False.

        Returns
        -------
        grid_points
            The grid points, shape (n_points, 3)

        """
        assert loaded_data is not None or mdata is not None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided"
        )
        assert loaded_data is None or mdata is None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided, not both"
        )

        X = self.var(FV.X)
        Y = self.var(FV.Y)
        H = self.var(FV.H)

        if mdata is not None:
            assert X in mdata and Y in mdata, (
                f"States '{self.name}': Missing coordinates '{X}' and/or '{Y}' in mdata, got {list(mdata.keys())}"
            )
            x = mdata[X]
            y = mdata[Y]

            if all_heights or height is None:
                assert H in mdata, (
                    f"States '{self.name}': Missing heights '{H}' in mdata, got {list(mdata.keys())}"
                )
                h = mdata[H]
                if height is None:
                    h = np.atleast_1d(np.max(h))
                elif all_heights:
                    raise ValueError(
                        f"States '{self.name}': Cannot specify both all_heights and height, got all_heights={all_heights}, height={height}"
                    )
            else:
                h = np.atleast_1d(height)

        else:
            assert loaded_data is not None
            assert X in loaded_data["coords"] and Y in loaded_data["coords"], (
                f"States '{self.name}': Missing coordinates '{X}' and/or '{Y}' in loaded_data, got {list(loaded_data['coords'].keys())}"
            )
            x = cast(np.ndarray, loaded_data["coords"][X])
            y = cast(np.ndarray, loaded_data["coords"][Y])

            if all_heights or height is None:
                assert H in loaded_data["coords"], (
                    f"States '{self.name}': Missing heights '{H}' in loaded_data, got {list(loaded_data['coords'].keys())}"
                )
                h = cast(np.ndarray, loaded_data["coords"][H])
                if height is None:
                    h = np.atleast_1d(np.max(h))
                elif all_heights:
                    raise ValueError(
                        f"States '{self.name}': Cannot specify both all_heights and height, got all_heights={all_heights}, height={height}"
                    )
            else:
                h = np.atleast_1d(height)

        nx = len(x)
        ny = len(y)
        nh = len(h)
        gpts = np.zeros((nx, ny, nh, 3), dtype=x.dtype)
        gpts[..., 0] = x[:, None, None]
        gpts[..., 1] = y[None, :, None]
        gpts[..., 2] = h[None, None, :]
        gpts = gpts.reshape(nx * ny * nh, 3)

        return gpts

    def _get_calc_data(
        self, mdata: MData, fdata: FData
    ) -> tuple[dict[tuple[str, ...], tuple[list[str], np.ndarray]], np.ndarray | None]:
        """
        Gathers data for calculations.

        Parameters
        ----------
        mdata
            The mdata object
        fdata
            The fdata object

        Returns
        -------
        data
            The extracted data, keys are dimension tuples,
            values are tuples (variables, data_array)
            where variables are the variable names, and
            data_array contains the data values,
            the last dimension corresponds to the variables
        weights
            The weights array, if only state dependent, otherwise
            weights are among data. Shape: (n_states,)

        """
        # prepare
        assert FC.STATE in self._cmap, (
            f"States '{self.name}': States coordinate '{FC.STATE}' not in cmap {self._cmap}"
        )
        n_states = mdata.n_states
        metadata = mdata.extra_data[self.META]
        assert metadata is not None
        data_keys = metadata["data_keys"]

        # extract data from mdata
        weights = mdata[FV.WEIGHT] if FV.WEIGHT in mdata else None
        data: dict[tuple[str, ...], tuple[list[str], np.ndarray]] = {}
        for DATA in data_keys:
            dims = mdata.dims[DATA]
            vrs: list[str] = (
                mdata[dims[-1]]
                if isinstance(mdata[dims[-1]], list)
                else mdata[dims[-1]].tolist()
            )
            dms = []
            for c in dims[:-1]:
                c0 = self.unvar(c) if c not in [FC.STATE, FC.TURBINE] else c
                dms.append(c0)
            dims_new = cast(tuple[str, ...], tuple(dms + [dims[-1]]))
            data[dims_new] = (vrs, mdata[DATA].copy())

        # adjust turbine order for purely turbine dependent data:
        mvd = []
        for dims in data.keys():
            if FC.STATE not in dims and FC.TURBINE in dims:
                assert dims[0] == FC.TURBINE, (
                    f"States '{self.name}': Turbine dimension must be the first dimension if state independent, but got {dims}"
                )
                mvd.append(dims)
        if len(mvd) > 0:
            ssel = fdata[FV.ORDER_SSEL].astype(config.dtype_int)
            order = fdata[FV.ORDER].astype(config.dtype_int)
            for dims0 in mvd:
                vrs, d0 = data.pop(dims0)
                dims = (FC.STATE,) + dims0
                d = np.zeros((n_states,) + d0.shape, dtype=d0.dtype)
                d[:] = d0[None, ...]
                d = d[ssel, order, ...]
                del d0

                if dims in data:
                    vrs = data[dims][0] + vrs
                    d = np.concatenate((data[dims][1], d), axis=-1)
                data[dims] = (vrs, d)

        return data, weights

    def get_interpolation_grid_data(
        self, mdata: MData, idims: list[str]
    ) -> tuple[np.ndarray, ...]:
        """
        Extracts interpolation grid data from chunk model data.

        Parameters
        ----------
        mdata
            The model data
        idims
            The dimensions for interpolation, e.g. ['x', 'y', 'height']

        Returns
        -------
        gpts
            One 1D array per interpolation dimension.

        """
        gpts = []
        for c in idims:
            cc = self.var(c) if c not in [FC.STATE, FC.TURBINE] else c
            assert cc in mdata, (
                f"States '{self.name}': Missing coordinate '{cc}' in mdata, got {list(mdata.keys())}"
            )
            gpts.append(mdata[cc])
        return tuple(gpts)

    def interpolate_data(
        self,
        mdata: MData,
        idims: list[str],
        d: np.ndarray,
        pts: np.ndarray,
        vrs: list[str],
        state_indices: np.ndarray | None = None,
        gpts: tuple[np.ndarray, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Interpolates data to points.

        Parameters
        ----------
        mdata
            The model data
        idims
            The input dimensions, e.g. ['x', 'y', 'height']
        d
            The data array, with shape (n1, n2, ..., nv)
            where ni represents the dimension sizes of the ordered
            icoords keys, and nv is the number of variables
        pts
            The points to interpolate to, with shape (n_pts, n_idims)
        vrs
            The variable names, length nv
        state_indices
            The indices of the states, with shape (n_states,)
        gpts
            One 1D array per dimension, or a 2D array with shape
            (n_points, n_dims), or None to extract the grid points from mdata.

        Returns
        -------
        d_interp
            The interpolated data array with shape (n_pts, nv)

        """
        if gpts is None:
            gpts = self.get_interpolation_grid_data(mdata, idims)
        else:
            assert isinstance(gpts, (tuple, list)) and len(gpts) == len(idims), (
                f"States '{self.name}': gpts must be a tuple or list of length {len(idims)}, got {type(gpts)} with length {len(gpts) if isinstance(gpts, (tuple, list)) else 'N/A'}"
            )

        assert (
            isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[1] == len(idims)
        ), (
            f"States '{self.name}': pts must be a 2D numpy array with shape (n_pts, {len(idims)}) for idims = {idims}, got {pts.shape}"
        )
        assert (
            isinstance(d, np.ndarray)
            and d.ndim >= len(gpts) + 1
            and d.shape[-1] == len(vrs)
            and d.shape[: len(gpts)] == tuple(len(g) for g in gpts)
        ), (
            f"States '{self.name}': d must be a numpy array with shape {tuple([len(g) for g in gpts] + ['...', len(vrs)])}, got {d.shape}"
        )

        try:
            ipars: dict[str, bool | None] = dict(bounds_error=True, fill_value=None)
            ipars.update(self.interp_pars)  # type: ignore[arg-type]
            d = interpn(gpts, d, pts, **ipars)
        except ValueError as e:
            print(f"\nStates '{self.name}': Interpolation error")
            print(f"INTERPOLATION DIMENSIONS: {idims}")
            print(
                "DATA BOUNDS:",
                [float(np.min(g)) for g in gpts],
                [float(np.max(g)) for g in gpts],
            )
            print(
                "EVAL BOUNDS:",
                [float(np.min(p)) for p in pts.T],
                [float(np.max(p)) for p in pts.T],
            )
            print(
                "INSIDE     :",
                [
                    float(np.min(p)) >= float(np.min(gpts[i]))
                    and float(np.max(p)) <= float(np.max(gpts[i]))
                    for i, p in enumerate(pts.T)
                ],
            )
            print(
                "\nMaybe you want to try the option 'bounds_error=False' in 'interp_pars'? This will extrapolate the data.\n"
            )
            raise e

        return d

    def calculate(  # type: ignore[override]
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values with shape
            (n_states, n_targets, n_tpoints)

        """

        # load data on the fly, if necessary:
        super().calculate(algo, mdata, fdata, tdata)

        # prepare
        self.ensure_output_vars(algo, tdata)
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        assert n_states is not None
        assert n_targets is not None
        assert n_tpoints is not None
        points = tdata[FC.TARGETS].reshape(n_states, n_targets * n_tpoints, 3)
        n_pts = points.shape[1]
        times = mdata[FC.STATE]
        sinds: np.ndarray = np.arange(n_states, dtype=config.dtype_int)

        # get data for calculation
        data, weights = self._get_calc_data(mdata, fdata)

        # check if points are state dependent
        _points_data: dict[str, Any] | None = None

        def _analyze_points(
            has_p: bool,
            has_h: bool,
            hcoords: dict[str, np.ndarray] | None = None,
        ) -> dict[str, Any]:
            """Helper function for points analysis."""
            nonlocal _points_data

            if _points_data is None:
                pmin = np.min(points, axis=0)
                pmax = np.max(points, axis=0)
                _points_data = {}
                _points_data["pmin"] = pmin
                _points_data["pmax"] = pmax
            else:
                pmin = _points_data["pmin"]
                pmax = _points_data["pmax"]

            if has_p and "points_vary" not in _points_data:
                if (
                    hcoords is not None
                    and FC.TURBINE in hcoords
                    and len(hcoords[FC.TURBINE].shape) == 3
                ):
                    _points_data["up"] = points
                    _points_data["points_vary"] = False
                elif np.any(pmax - pmin > 1e-4):
                    _points_data["up"], _points_data["up2p"] = np.unique(
                        points.reshape(n_states * n_pts, 3), axis=0, return_inverse=True
                    )
                    _points_data["points_vary"] = True
                else:
                    _points_data["up"] = points[0]
                    _points_data["up2p"] = None
                    _points_data["points_vary"] = False

            if has_h and "heights_vary" not in _points_data:
                if np.any(pmax[:, 2] - pmin[:, 2] > 1e-4):
                    _points_data["uh"], _points_data["uh2h"] = np.unique(
                        points[:, :, 2].reshape(n_states * n_pts), return_inverse=True
                    )
                    _points_data["heights_vary"] = True
                else:
                    _points_data["uh"] = points[0, :, 2]
                    _points_data["uh2h"] = None
                    _points_data["heights_vary"] = False

            return _points_data

        # interpolate data to points:
        out: dict[str, np.ndarray] = {}
        for dims, (vrs, d) in data.items():
            # replace (WD, WS) by (U, V):
            iwd = None
            if FV.WD in vrs or FV.WS in vrs:
                assert FV.WD in vrs and (FV.WS in vrs or FV.WS in self.fixed_vars), (
                    f"States '{self.name}': Missing '{FV.WD}' or '{FV.WS}' in data variables {vrs} for dimensions {dims}"
                )
                assert FV.U not in vrs and FV.U not in vrs, (
                    f"States '{self.name}': Cannot have '{FV.WD}', '{FV.WS}' and  '{FV.U}', '{FV.V}' in data variables {vrs} for dimensions {dims}"
                )
                iwd = vrs.index(FV.WD)
                iws = vrs.index(FV.WS)
                ws = d[..., iws] if FV.WS in vrs else self.fixed_vars[FV.WS]
                d[..., [iwd, iws]] = wd2uv(d[..., iwd], ws, axis=-1)
                del ws
            elif FV.U in vrs or FV.V in vrs:
                assert FV.U in vrs and FV.V in vrs, (
                    f"States '{self.name}': Missing '{FV.U}' or '{FV.V}' in variables {vrs} for dims {dims}"
                )
                iwd = vrs.index(FV.U)
                iws = vrs.index(FV.V)

            # move state dimension to second last position:
            if dims[0] == FC.STATE:
                # and not (
                #    FC.TURBINE in hcoords and len(hcoords[FC.TURBINE].shape) == 3
                # ):
                d = np.moveaxis(d, 0, -2)
                dims = dims[1:-1] + (FC.STATE,) + (dims[-1],)
                idims = list(dims[:-2])
            else:
                idims = list(dims[:-1])

            # interpolate data:
            n_vrs = len(vrs)
            if len(idims) > 0:
                # prepare points:
                pts = []
                has_p = (
                    FV.X in idims
                    or FV.Y in idims
                    or FC.POINT in idims
                    or FC.TURBINE in idims
                )
                has_h = FV.H in idims or FC.TURBINE in idims
                for c in idims.copy():
                    if c in [FV.X, FV.Y, FV.H]:
                        points_data = _analyze_points(has_p, has_h)
                        if c in [FV.X, FV.Y]:
                            i = 0 if c == FV.X else 1
                            pts.append(points_data["up"][:, i])
                        elif has_p:
                            pts.append(points_data["up"][:, 2])
                        else:
                            pts.append(points_data["uh"])
                    elif c == FC.POINT:
                        points_data = _analyze_points(has_p, has_h)
                        pts.append(points_data["up"][..., 0])
                        pts.append(points_data["up"][..., 1])
                    elif c == FC.TURBINE:
                        points_data = _analyze_points(has_p, has_h)
                        pts.append(points_data["up"][..., 0])
                        pts.append(points_data["up"][..., 1])
                        if points_data["up"].shape[-1] >= 3:
                            pts.append(points_data["up"][..., 2])
                    elif c == FC.STATE:
                        idims.remove(FC.STATE)
                    else:
                        raise NotImplementedError(
                            f"States '{self.name}': Unsupported dimension '{c}' in {dims} for interpolation of variables {vrs}"
                        )
                interpolation_points: np.ndarray | None = (
                    np.stack(pts, axis=-1) if len(pts) > 0 else None
                )

                # interpolate:
                assert interpolation_points is not None
                d = self.interpolate_data(
                    mdata, idims, d, interpolation_points, vrs, times
                )

                # move state dimension back to front:
                if dims[0] == FC.STATE:
                    pass
                elif FC.STATE in dims:
                    dims = (FC.STATE,) + dims[:-2] + (dims[-1],)
                    d = np.moveaxis(d, -2, 0)
                else:
                    d = d[None, ...]

                # reconstruct time varying pts:
                if has_p and points_data["points_vary"]:
                    shp = d.shape[0:1] + (n_states, n_pts) + d.shape[2:]
                    d = d[:, points_data["up2p"], :].reshape(shp)
                    if FC.STATE in dims:
                        d = d[sinds, sinds, ...]
                    else:
                        d = d[0, ...]
                elif has_h and points_data["heights_vary"]:
                    shp = d.shape[0:1] + (n_states, n_pts) + d.shape[2:]
                    d = d[:, points_data["uh2h"], :].reshape(shp)
                    if FC.STATE in dims:
                        d = d[sinds, sinds, ...]
                    else:
                        d = d[0, ...]
                del pts

            # case no interpolation needed:
            else:
                # reshape to include states and points dimensions:
                if dims[0] == FC.STATE:
                    d = d[:, None, :]
                else:
                    d = d[None, None, :]

            # translate (U, V) into (WD, WS):
            if iwd is not None:
                if FV.WD not in vrs:
                    vrs = vrs.copy()
                    vrs[iwd] = FV.WD
                    vrs[iws] = FV.WS
                uv = d[..., [iwd, iws]]
                d[..., iwd] = uv2wd(uv)
                d[..., iws] = np.linalg.norm(uv, axis=-1)
                del uv

            # broadcast if needed:
            if d.shape != (n_states, n_pts, n_vrs):
                tmp = d
                d = np.zeros((n_states, n_pts, n_vrs), dtype=config.dtype_double)
                d[:] = tmp
                del tmp

            # set output:
            for i, v in enumerate(vrs):
                out[v] = d[..., i]

        # set fixed variables:
        for v, fixed_value in self.fixed_vars.items():
            out[v] = np.full((n_states, n_pts), fixed_value, dtype=config.dtype_double)

        # add weights:
        if weights is not None:
            tdata[FV.WEIGHT] = weights[:, None, None]
        elif FV.WEIGHT in out:
            tdata[FV.WEIGHT] = out.pop(FV.WEIGHT).reshape(
                n_states, n_targets, n_tpoints
            )
        else:
            assert self._N is not None and self._N > 0
            tdata[FV.WEIGHT] = np.full(
                (mdata.n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
            )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        # reshape results:
        results: dict[str, np.ndarray] = {
            v: data_array.reshape(n_states, n_targets, n_tpoints)
            for v, data_array in out.items()
        }
        del out

        # convert TKE to TI if needed:
        if FV.TI in self.ovars and FV.TI not in results:
            assert FV.WS in results, (
                f"States '{self.name}': Cannot calculate {FV.TI} without {FV.WS}"
            )
            assert FV.TKE in results or FV.TKE in self.ovars, (
                f"States '{self.name}': Cannot calculate {FV.TI} without {FV.TKE}"
            )
            if FV.TKE not in self.ovars:
                tke = np.maximum(results.pop(FV.TKE), 1e-10)
            else:
                tke = np.maximum(results[FV.TKE], 1e-10)
            ws = results[FV.WS]
            assert np.all(ws > 0.0), (
                f"States '{self.name}': Cannot calculate {FV.TI} from {FV.TKE}, found zeros or negative values in {FV.WS}"
            )
            results[FV.TI] = np.sqrt(1.5 * tke) / ws

        # compute air density if needed:
        if FV.RHO in self.ovars and FV.RHO not in results:
            assert FV.p in results, (
                f"States '{self.name}': Cannot calculate {FV.RHO} without {FV.p}"
            )
            assert FV.T in results, (
                f"States '{self.name}': Cannot calculate {FV.RHO} without {FV.T}"
            )
            if FV.p not in self.ovars:
                p = results.pop(FV.p)
            else:
                p = results[FV.p]
            if FV.T not in self.ovars:
                T = results.pop(FV.T)
            else:
                T = results[FV.T]
            assert not np.any(T <= 0.0), (
                f"States '{self.name}': Cannot calculate {FV.RHO}, found zeros or negative values in {FV.T}"
            )
            results[FV.RHO] = p / (FC.Rd * T)

        return results
