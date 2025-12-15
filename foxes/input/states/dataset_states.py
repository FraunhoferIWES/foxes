import numpy as np
import pandas as pd
import xarray as xr
from copy import copy, deepcopy
from scipy.interpolate import interpn

from foxes.core import States, get_engine
from foxes.utils import import_module
from foxes.data import STATES, StaticData
from foxes.utils.wind_dir import uv2wd, wd2uv
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC


def _read_nc_file(
    fpath,
    coords,
    vars,
    nc_engine,
    sel,
    isel,
    minimal,
    drop_vars=None,
    check_input_nans=True,
    preprocess=None,
):
    """Helper function for nc file reading"""
    with xr.open_dataset(fpath, drop_variables=drop_vars, engine=nc_engine) as data:
        for c in coords:
            if c is not None and c not in data.sizes:
                raise KeyError(
                    f"Missing coordinate '{c}' in file {fpath}, got: {list(data.sizes.keys())}"
                )
        if preprocess is not None:
            data = preprocess(data)
        if minimal:
            data = data[coords[0]].to_numpy()
        else:
            data = data[vars]
            data.attrs = {}
            if isel is not None and len(isel):
                isel = {c: s for c, s in isel.items() if c in data.sizes}
                data = data.isel(**isel)
            if sel is not None and len(sel):
                sel = {c: s for c, s in sel.items() if c in data.sizes}
                data = data.sel(**sel)
            assert min(data.sizes.values()) > 0, (
                f"States: No data in file {fpath}, isel={isel}, sel={sel}, resulting sizes={data.sizes}"
            )
            if check_input_nans:
                for v, d in data.data_vars.items():
                    sel = np.isnan(d.to_numpy())
                    if sel.any():
                        i = tuple([j[0] for j in np.where(sel)])
                        print("\n\nError: NaN data found in input data:")
                        print(f"  File: {fpath}\n")
                        print(f"  Variable: {v}")
                        for ic, c in enumerate(d.dims):
                            print(f"  {c}: {data[c].to_numpy()[i[ic]]}")
                        print("\n\n")
                        raise ValueError(
                            f"States: NaN data found in input data for variable '{v}' with dims {d.dims} in file {fpath} at index {i}"
                        )
    return data


class DatasetStates(States):
    """
    Abstract base class for heterogeneous ambient states that
    are based on data from NetCDF files or an xarray Dataset.

    Attributes
    ----------
    data_source: str or xarray.Dataset
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    ovars: list of str
        The output variables
    var2ncvar: dict
        Mapping from variable names to variable names
        in the nc file
    fixed_vars: dict
        Uniform values for output variables, instead
        of reading from data
    load_mode: str
        The load mode, choices: preload, lazy, fly.
        preload loads all data during initialization,
        lazy lazy-loads the data using dask, and fly
        reads only states index and weights during initialization
        and then opens the relevant files again within
        the chunk calculation
    time_format: str
        The datetime parsing format string
    sel: dict, optional
        Subset selection via xr.Dataset.sel()
    isel: dict, optional
        Subset selection via xr.Dataset.isel()
    weight_factor: float
        The factor to multiply the weights with
    check_times: bool
        Whether to check the time coordinates for consistency
    check_input_nans: bool
        Whether to check input data for NaNs
    preprocess_nc: callable, optional
        A function to preprocess the netcdf Dataset before use
    interp_pars: dict
        Additional parameters the interpolation

    :group: input.states

    """

    def __init__(
        self,
        data_source,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        load_mode="preload",
        time_format=None,
        sel=None,
        isel=None,
        weight_factor=None,
        check_times=True,
        check_input_nans=True,
        preprocess_nc=None,
        interp_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or xarray.Dataset
            The data or the file search pattern, should end with
            suffix '.nc'. One or many files.
        output_vars: list of str
            The output variables
        var2ncvar: dict, optional
            Mapping from variable names to variable names
            in the nc file
        fixed_vars: dict, optional
            Uniform values for output variables, instead
            of reading from data
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculation
        time_format: str, optional
            The datetime parsing format string
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        weight_factor: float, optional
            The factor to multiply the weights with
        check_times: bool
            Whether to check the time coordinates for consistency
        check_input_nans: bool
            Whether to check input data for NaNs, otherwise NaNs are removed
        preprocess_nc: callable, optional
            A function to preprocess the netcdf Dataset before use
        interp_pars: dict, optional
            Additional parameters the interpolation
        kwargs: dict, optional
            Additional arguments for the base class

        """
        super().__init__(**kwargs)

        self.ovars = list(output_vars)
        self.fixed_vars = fixed_vars
        self.load_mode = load_mode
        self.var2ncvar = var2ncvar
        self.time_format = time_format
        self.sel = sel
        self.isel = isel
        self.weight_factor = weight_factor
        self.check_times = check_times
        self.check_input_nans = check_input_nans
        self.preprocess_nc = preprocess_nc
        self.interp_pars = interp_pars

        self._N = None
        self._inds = None
        self.__data_source = data_source

    @property
    def data_source(self):
        """
        The data source

        Returns
        -------
        s: object
            The data source

        """
        if self.load_mode in ["preload", "fly"] and self.running:
            raise ValueError(
                f"States '{self.name}': Cannot access data_source while running for load mode '{self.load_mode}'"
            )
        return self.__data_source

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
        data = {}
        for v in variables:
            w = self.var2ncvar.get(v, v)
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
                    f"States '{self.name}': Variable '{w}' not found in data source '{self.data_source}', available variables: {list(ds.data_vars)}"
                )

        coords = {v: ds[c].to_numpy() for v, c in cmap.items() if c in ds.coords}

        if verbosity > 1:
            if len(coords):
                print(f"\n{self.name}: Coordinate ranges")
                for c, d in coords.items():
                    print(f"  {c}: {np.min(d)} --> {np.max(d)}")
            print(f"\n{self.name}: Data ranges")
            for v, d in data.items():
                nn = np.sum(np.isnan(d))
                print(
                    f"  {v}: {np.nanmin(d)} --> {np.nanmax(d)}, nans: {nn} ({100 * nn / len(d.flat):.2f}%)"
                )

        return coords, data

    def _get_data(self, ds, cmap, variables, verbosity=0):
        """
        Gets the data from the Dataset and prepares it for calculations.

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
        coords, data0 = self._read_ds(ds, cmap, variables, verbosity=verbosity)

        weights = None
        if FV.WEIGHT in variables:
            assert FV.WEIGHT in data0, (
                f"States '{self.name}': Missing weights variable '{FV.WEIGHT}' in data, found {sorted(list(data0.keys()))}"
            )
            if self.weight_factor is not None:
                data0[FV.WEIGHT][1] *= self.weight_factor
            if data0[FV.WEIGHT][0] == (FC.STATE,):
                weights = data0.pop(FV.WEIGHT)[1]

        data = {}  # dim: [DATA key, variables, data array]
        for v, (dims, d) in data0.items():
            if dims not in data:
                i = len(data)
                data[dims] = [self.var(f"data{i}"), [], []]
            data[dims][1].append(v)
            data[dims][2].append(d)
        for dims in data.keys():
            data[dims][2] = np.stack(data[dims][2], axis=-1)
        data = {
            tuple(list(dims) + [f"vars{i}"]): d
            for i, (dims, d) in enumerate(data.items())
        }
        return coords, data, weights

    def _find_xy_bounds(self, algo, bounds_extra_space):
        """Helper function to determine x/y bounds with extra space."""
        return algo.farm.get_xy_bounds(extra_space=bounds_extra_space, algo=algo)

    def preproc_first(
        self, algo, data, cmap, vars, bounds_extra_space, height_bounds, verbosity=0
    ):
        """
        Preprocesses the first file.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: xarray.Dataset
            The dataset to preprocess
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        vars: list
            The list of variable names
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple, optional
            The (h_min, h_max) height bounds in m. Defaults to H +/-
        verbosity: int
            The verbosity level, 0 = silent

        """

        # find vertical bounds:
        if FV.H in cmap:
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
            self._heights = data[cmap[FV.H]].to_numpy()
            if (
                np.min(self._heights) > height_bounds[0]
                or np.max(self._heights) < height_bounds[1]
            ):
                raise ValueError(
                    f"States '{self.name}': Height bounds {height_bounds} m are outside of data height range {np.min(self._heights)} - {np.max(self._heights)} m"
                )
            ch = cmap[FV.H]
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
            assert FV.X in cmap, (
                f"States '{self.name}': x coordinate '{FV.X}' not in cmap {cmap}"
            )
            assert FV.Y in cmap, (
                f"States '{self.name}': y coordinate '{FV.Y}' not in cmap {cmap}"
            )
            xy_min, xy_max = self._find_xy_bounds(algo, bounds_extra_space)
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting xy to bounds {xy_min} - {xy_max}"
                )
            for v, i in zip((FV.X, FV.Y), (0, 1)):
                if self.isel is None or cmap[v] not in self.isel:
                    x0, x1 = xy_min[i], xy_max[i]
                    x = data[cmap[v]].to_numpy()
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
                    self.isel.update({cmap[v]: slice(i0, i1 + 1)})
                if verbosity > 0:
                    hv = data[cmap[v]].isel({cmap[v]: self.isel[cmap[v]]}).to_numpy()
                    print(
                        f"States '{self.name}': Selected {cmap[v]} = {hv[0]} ... {hv[-1]} ({len(hv)} points)"
                    )

    def __preload(
        self,
        algo,
        cmap,
        bounds_extra_space,
        height_bounds,
        verbosity=0,
    ):
        """Helper function for preloading data."""

        assert FC.STATE in cmap, (
            f"States '{self.name}': States coordinate '{FC.STATE}' not in cmap {cmap}"
        )
        states_coord = cmap[FC.STATE]

        if not isinstance(self.data_source, xr.Dataset):
            # check static data:
            fpath = get_input_path(self.data_source)
            if "*" not in str(self.data_source):
                if not fpath.is_file():
                    fpath = StaticData().get_file_path(
                        STATES, fpath.name, check_raw=False
                    )

            # find files:
            prt = fpath.resolve().parent
            glb = fpath.name
            while "*" in str(prt):
                glb = prt.name + "/" + glb
                prt = prt.parent
            files = sorted(list(prt.glob(glb)))
            coords = list(cmap.values())
            vars = [self.var2ncvar.get(v, v) for v in self.variables]

            # pre-process first file:
            fpath = files[0]
            if verbosity > 0:
                print(f"States '{self.name}': Preprocessing first file", fpath.name)
            with xr.open_dataset(fpath, engine=config.nc_engine) as data_first:
                self.drop_vars = [
                    v for v in data_first.data_vars if v not in coords + vars
                ]
                if len(self.drop_vars) > 0 and verbosity > 0:
                    print(f"States '{self.name}': Keeping variables  {vars}")
                    print(f"States '{self.name}': Dropping variables {self.drop_vars}")
                if self.preprocess_nc is not None:
                    data_first = self.preprocess_nc(data_first)
                self.preproc_first(
                    algo,
                    data=data_first,
                    cmap=cmap,
                    vars=vars,
                    bounds_extra_space=bounds_extra_space,
                    height_bounds=height_bounds,
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

            self.__data_source = get_engine().map(
                _read_nc_file,
                files,
                coords=coords,
                vars=vars,
                nc_engine=config.nc_engine,
                isel=self.isel,
                sel=self.sel,
                minimal=self.load_mode == "fly",
                drop_vars=self.drop_vars,
                check_input_nans=self.check_input_nans,
                preprocess=self.preprocess_nc,
            )

            if self.load_mode in ["preload", "lazy"]:
                self._input_sizes = [
                    ds.sizes[states_coord] for ds in self.__data_source
                ]
                if self.load_mode == "lazy":
                    try:
                        self.__data_source = [ds.chunk() for ds in self.__data_source]
                    except (ModuleNotFoundError, ValueError) as e:
                        import_module("dask")
                        raise e
                if len(self.__data_source) == 1:
                    self.__data_source = self.__data_source[0]
                else:
                    self.__data_source = xr.concat(
                        self.__data_source,
                        dim=states_coord,
                        coords="minimal",
                        data_vars="minimal",
                        compat="equals",
                        join="exact",
                        combine_attrs="drop",
                    )
                if self.load_mode == "preload":
                    self.__data_source.load()
                self._inds = self.__data_source[states_coord].to_numpy()
                self._N = len(self._inds)

            elif self.load_mode == "fly":
                self._inds = self.__data_source
                self.__data_source = fpath
                self._files_maxi = {f: len(inds) for f, inds in zip(files, self._inds)}
                self._input_sizes = list(self._files_maxi.values())
                self._inds = np.concatenate(self._inds, axis=0)
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
            self._inds = self.data_source[states_coord].to_numpy()
            self._N = len(self._inds)

        # make sure state indices are sorted ascending:
        def _is_sorted(a):
            return np.all(a[:-1] <= a[1:])

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

        return self.__data_source

    def gen_states_split_size(self):
        """
        Generator for suggested states split sizes for output writing.

        Yields
        ------
        split_size: int or None
            The suggested split size, or None for no splitting

        """
        for size in self._input_sizes:
            yield size

    def load_data(
        self,
        algo,
        cmap,
        variables,
        bounds_extra_space=None,
        height_bounds=None,
        verbosity=0,
    ):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        variables: list of str
            The variables to extract from the Dataset
        bounds_extra_space: float, optional
            The extra space in meters to add to the horizontal wind farm bounds
        height_bounds: tuple, optional
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        # preload data:
        self.__preload(
            algo,
            cmap,
            bounds_extra_space,
            height_bounds,
            verbosity=verbosity,
        )

        idata = super().load_data(algo, verbosity)

        if self.load_mode == "preload":
            self._coords, data, w = self._get_data(
                self.data_source, cmap, variables, verbosity
            )

            if FC.STATE in self._coords:
                idata["coords"][FC.STATE] = self._coords.pop(FC.STATE)
            else:
                del idata["coords"][FC.STATE]
            if w is not None:
                idata["data_vars"][FV.WEIGHT] = ((FC.STATE,), w)

            vmap = {FC.STATE: FC.STATE}
            self._data_state_keys = []
            self._data_nostate = {}
            for dims, d in data.items():
                dms = tuple([vmap.get(c, self.var(c)) for c in dims])
                if FC.STATE in dims:
                    self._data_state_keys.append(d[0])
                    idata["coords"][dms[-1]] = d[1]
                    idata["data_vars"][d[0]] = (dms, d[2])
                else:
                    self._data_nostate[dims] = (d[1], d[2])
            del data

        return idata

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict, optional
            Large data stash, this function adds data here, if given.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
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
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict, optional
            Reconstruct model data from this stash, if given.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data = data_stash[self.name]
            self._inds = data.pop("inds")

            if self.load_mode == "preload":
                self.__data_source = data.pop("data_source")

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

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        if self.running:
            raise ValueError(f"States '{self.name}': Cannot access index while running")
        return self._inds

    def get_calc_data(self, mdata, cmap, variables):
        """
        Gathers data for calculations.

        Call this function from the calculate function of the
        derived class.

        Parameters
        ----------
        mdata: foxes.core.MData
            The mdata object
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        variables: list of str
            The variables to extract from the Dataset

        Returns
        -------
        coords: dict
            keys: Foxes variable names, values: 1D coordinate value arrays
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
        # prepare
        assert FC.STATE in cmap, (
            f"States '{self.name}': States coordinate '{FC.STATE}' not in cmap {cmap}"
        )
        states_coord = cmap[FC.STATE]
        n_states = mdata.n_states

        # case preload
        if self.load_mode == "preload":
            coords = self._coords
            weights = mdata[FV.WEIGHT] if FV.WEIGHT in mdata else None
            data = deepcopy(self._data_nostate)
            for DATA in self._data_state_keys:
                dims = mdata.dims[DATA]
                vrs = mdata[dims[-1]].tolist()
                dms = tuple(
                    [self.unvar(c) if c != FC.STATE else FC.STATE for c in dims[:-1]]
                    + [dims[-1]]
                )
                data[dms] = (vrs, mdata[DATA].copy())

        # case lazy
        elif self.load_mode == "lazy":
            i0 = mdata.states_i0(counter=True)
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({states_coord: s}).load()
            coords, data, weights = self._get_data(ds, cmap, variables, verbosity=0)
            data = {dims: (d[1], d[2]) for dims, d in data.items()}
            del ds

        # case fly
        elif self.load_mode == "fly":
            vars = [self.var2ncvar.get(v, v) for v in variables]
            i0 = mdata.states_i0(counter=True)
            i1 = i0 + n_states
            j0 = 0
            data = []
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

                        data.append(
                            _read_nc_file(
                                fpath,
                                coords=list(cmap.values()),
                                vars=vars,
                                nc_engine=config.nc_engine,
                                isel=isel,
                                sel=self.sel,
                                minimal=False,
                                drop_vars=self.drop_vars,
                                check_input_nans=self.check_input_nans,
                                preprocess=self.preprocess_nc,
                            )
                        )

                        i0 += b - a
                    j0 = j1

            assert i0 == i1, (
                f"States '{self.name}': Missing states for load_mode '{self.load_mode}': (i0, i1) = {(i0, i1)}"
            )
            if len(data) == 1:
                data = data[0]
            else:
                data = xr.concat(
                    data,
                    dim=states_coord,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="exact",
                    combine_attrs="drop",
                )
            coords, data, weights = self._get_data(data, cmap, variables, verbosity=0)
            data = {dims: (d[1], d[2]) for dims, d in data.items()}

        else:
            raise KeyError(
                f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
            )

        return coords, data, weights

    def interpolate_data(self, idims, icrds, d, pts, vrs, times):
        """
        Interpolates data to points.

        This function should be implemented in derived classes.

        Parameters
        ----------
        idims: list of str
            The input dimensions, e.g. [x, y, height]
        icrds: list of numpy.ndarray
            The input coordinates, each with shape (n_i,)
            where n_i is the number of grid points in dimension i
        d: numpy.ndarray
            The data array, with shape (n1, n2, ..., nv)
            where ni represents the dimension sizes and
            nv is the number of variables
        pts: numpy.ndarray
            The points to interpolate to, with shape (n_pts, n_idims)
        vrs: list of str
            The variable names, length nv
        times: numpy.ndarray
            The time coordinates of the states, with shape (n_states,)
        Returns
        -------
        d_interp: numpy.ndarray
            The interpolated data array with shape (n_pts, nv)

        """
        gvars = tuple(icrds)
        try:
            ipars = dict(bounds_error=True, fill_value=None)
            ipars.update(self.interp_pars)
            d = interpn(gvars, d, pts, **ipars)
        except ValueError as e:
            print(f"\nStates '{self.name}': Interpolation error")
            print(f"INPUT VARS: {idims}")
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
                "INSIDE     :",
                [
                    float(np.min(p)) >= float(np.min(gvars[i]))
                    and float(np.max(p)) <= float(np.max(gvars[i]))
                    for i, p in enumerate(pts.T)
                ],
            )
            print(
                "\nMaybe you want to try the option 'bounds_error=False' in 'interp_pars'? This will extrapolate the data.\n"
            )
            raise e

        return d

    def _update_dims(self, dims, coords, vrs, d):
        """Helper function for dimension adjustment, if needed"""
        return dims, coords

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
        points = tdata[FC.TARGETS].reshape(n_states, n_targets * n_tpoints, 3)
        n_pts = points.shape[1]
        times = mdata[FC.STATE]

        # get data for calculation
        coords, data, weights = self.get_calc_data(mdata, self._cmap, self.variables)
        coords[FC.STATE] = np.arange(n_states, dtype=config.dtype_int)

        # check if points are state dependent
        _points_data = None

        def _analyze_points(has_p, has_h):
            """Helper function for points analysis."""
            nonlocal _points_data

            if _points_data is None:
                pmin = np.min(points, axis=(0, 1))
                pmax = np.max(points, axis=(0, 1))
                _points_data = {}
                _points_data["pmin"] = pmin
                _points_data["pmax"] = pmax
            else:
                pmin = _points_data["pmin"]
                pmax = _points_data["pmax"]

            if has_p and "points_vary" not in _points_data:
                if np.max(pmax - pmin) > 1e-4:
                    _points_data["up"], _points_data["up2p"] = np.unique(
                        points.reshape(n_states * n_pts, 3), axis=0, return_inverse=True
                    )
                    _points_data["points_vary"] = True
                else:
                    _points_data["up"] = points[0]
                    _points_data["up2p"] = None
                    _points_data["points_vary"] = False

            if has_h and "heights_vary" not in _points_data:
                if np.max(pmax[2] - pmin[2]) > 1e-4:
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
        out = {}
        for dims, (vrs, d) in data.items():
            # update dims, if necessary:
            dims, hcoords = self._update_dims(dims, coords, vrs, d)

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
                has_p = FV.X in idims or FV.Y in idims or FC.POINT in idims
                has_h = FV.H in idims
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
                        pts.append(points_data["up"][:, 0])
                        pts.append(points_data["up"][:, 1])
                        if hcoords[FC.POINT].shape[1] == 3:
                            pts.append(points_data["up"][:, 2])
                    elif c == FC.STATE:
                        idims.remove(FC.STATE)
                    else:
                        raise NotImplementedError(
                            f"States '{self.name}': Unsupported dimension '{c}' in {dims} for interpolation of variables {vrs}"
                        )
                pts = np.stack(pts, axis=-1)

                # interpolate:
                icrds = [hcoords[c] for c in idims]
                d = self.interpolate_data(idims, icrds, d, pts, vrs, times)

                # move state dimension back to front:
                if FC.STATE in dims:
                    dims = (FC.STATE,) + dims[:-2] + (dims[-1],)
                    d = np.moveaxis(d, -2, 0)
                else:
                    d = d[None, ...]

                # reconstruct time varying pts:
                if has_p and points_data["points_vary"]:
                    shp = d.shape[0:1] + (n_states, n_pts) + d.shape[2:]
                    d = d[:, points_data["up2p"], :].reshape(shp)
                    if FC.STATE in dims:
                        d = d[hcoords[FC.STATE], hcoords[FC.STATE], ...]
                    else:
                        d = d[0, ...]
                elif has_h and points_data["heights_vary"]:
                    shp = d.shape[0:1] + (n_states, n_pts) + d.shape[2:]
                    d = d[:, points_data["uh2h"], :].reshape(shp)
                    if FC.STATE in dims:
                        d = d[hcoords[FC.STATE], hcoords[FC.STATE], ...]
                    else:
                        d = d[0, ...]
                del points_data, pts, icrds

            # case no interpolation needed:
            else:
                # reshape to include states and points dimensions:
                if dims[0] == FC.STATE:
                    d = d[:, None, :]
                else:
                    d = d[None, None, :]
            del hcoords

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
        for v, d in self.fixed_vars.items():
            out[v] = np.full((n_states, n_pts), d, dtype=config.dtype_double)

        # add weights:
        if weights is not None:
            tdata[FV.WEIGHT] = weights[:, None, None]
        elif FV.WEIGHT in out:
            tdata[FV.WEIGHT] = out.pop(FV.WEIGHT).reshape(
                n_states, n_targets, n_tpoints
            )
        else:
            tdata[FV.WEIGHT] = np.full(
                (mdata.n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
            )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        # reshape results:
        results = {v: d.reshape(n_states, n_targets, n_tpoints) for v, d in out.items()}
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
                tke = np.maximum(results.pop(FV.TKE), 0)
            else:
                tke = np.maximum(results[FV.TKE], 0)
            ws = results[FV.WS]
            assert not np.any(ws <= 0.0), (
                f"States '{self.name}': Cannot calculate {FV.TI}, found zeros in {FV.WS}"
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
