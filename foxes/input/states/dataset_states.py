import numpy as np
import pandas as pd
import xarray as xr
from abc import abstractmethod
from copy import copy, deepcopy

from foxes.core import States, get_engine
from foxes.utils import import_module
from foxes.data import STATES, StaticData
import foxes.variables as FV
import foxes.constants as FC
from foxes.config import config, get_input_path


def _read_nc_file(
    fpath,
    coords,
    vars,
    nc_engine,
    sel,
    isel,
    minimal,
):
    """Helper function for nc file reading"""
    data = xr.open_dataset(fpath, engine=nc_engine)
    for c in coords:
        if c is not None and c not in data.sizes:
            raise KeyError(
                f"Missing coordinate '{c}' in file {fpath}, got: {list(data.sizes.keys())}"
            )
    if minimal:
        return data[coords[0]].to_numpy()
    else:
        data = data[vars]
        data.attrs = {}
        if isel is not None and len(isel):
            data = data.isel(**isel)
        if sel is not None and len(sel):
            data = data.sel(**sel)
        assert min(data.sizes.values()) > 0, (
            f"States: No data in file {fpath}, isel={isel}, sel={sel}, resulting sizes={data.sizes}"
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

    :group: input.states

    """

    def __init__(
        self,
        data_source,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        load_mode="preload",
        time_format="%Y-%m-%d_%H:%M:%S",
        sel=None,
        isel=None,
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
        time_format: str
            The datetime parsing format string
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        weight_factor: float, optional
            The factor to multiply the weights with
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

    @abstractmethod
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
        gpts: tuple
            Grid point data extracted from the Dataset, like
            (x, y, h) or similar
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
        pass

    def _preload(self, algo, coords, filter_xy, verbosity=0):
        """Helper function for preloading data."""

        states_coord = coords[0]
        if not isinstance(self.data_source, xr.Dataset):

            # check static data:
            fpath = get_input_path(self.data_source)
            if "*" not in str(self.data_source):
                if not fpath.is_file():
                    fpath = StaticData().get_file_path(
                        STATES, fpath.name, check_raw=False
                    )

            # find bounds:
            if filter_xy is not None:
                x_coord = filter_xy["x_coord"]
                y_coord = filter_xy["y_coord"]
                bspace = filter_xy["bounds_extra_space"]
                assert x_coord in coords, f"States '{self.name}': x coordinate '{x_coord}' from filter_xy not in coords {coords}"
                assert y_coord in coords, f"States '{self.name}': y coordinate '{y_coord}' from filter_xy not in coords {coords}"

                #if bounds and self.x_coord is not None and self.x_coord not in self.sel:
                xy_min, xy_max = algo.farm.get_xy_bounds(
                    extra_space=bspace, algo=algo
                )
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Restricting to bounds {xy_min} - {xy_max}"
                    )
                if self.sel is None:
                    self.sel = {}
                self.sel.update(
                    {
                        x_coord: slice(xy_min[0], xy_max[1]),
                        y_coord: slice(xy_min[1], xy_max[1]),
                    }
                )

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
            files = sorted(list(fpath.resolve().parent.glob(fpath.name)))
            vars = [self.var2ncvar.get(v, v) for v in self.variables]
            self.__data_source = get_engine().map(
                _read_nc_file,
                files,
                coords=coords,
                vars=vars,
                nc_engine=config.nc_engine,
                isel=self.isel,
                sel=self.sel,
                minimal=self.load_mode == "fly",
            )
    
            if self.load_mode in ["preload", "lazy"]:
                if self.load_mode == "lazy":
                    try:
                        self.__data_source = [ds.chunk() for ds in self.__data_source]
                    except (ModuleNotFoundError, ValueError) as e:
                        import_module("dask")
                        raise e
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

        return self.__data_source

    def load_data(
        self, 
        algo, 
        coords, 
        variables,
        filter_xy=None,
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
        coords: list of str
            The relevant coordinate names in the original Dataset,
            e.g. (states_coord, x_coord, y_coord). The first entry
            is interpreted as the states coordinate.
        variables: list of str
            The variables to extract from the Dataset
        filter_xy: dict, optional
            Parameters for filtering the (x, y) coordinates. Expects keys
            x_coord, y_coord, bounds_extra_space; with resp values of types
            str, str, float.
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
        self._preload(algo, coords, filter_xy, verbosity=verbosity)

        idata = super().load_data(algo, verbosity)

        if self.load_mode == "preload":

            s, self._gpts, data, w = self._get_data(
                self.data_source, variables, verbosity)

            if s is not None:
                idata["coords"][FC.STATE] = s
            else:
                del idata["coords"][FC.STATE]
            if w is not None:
                idata["data_vars"][FV.WEIGHT] = ((FC.STATE,), w)

            self._data_state_keys = []
            self._data_nostate = {}
            for dims, d in data.items():
                if FC.STATE in dims:
                    self._data_state_keys.append(d[0])
                    idata["coords"][dims[-1]] = d[1]
                    idata["data_vars"][d[0]] = (dims, d[2])
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
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        data_stash[self.name] = dict(
            inds=self._inds,
        )
        del self._inds

        if self.load_mode == "preload":
            data_stash[self.name]["data_source"] = self.__data_source
            del self.__data_source

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
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

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

    def get_calc_data(self, mdata, coords, variables):
        """
        Gathers data for calculations.

        Call this function from the calculate function of the
        derived class.

        Parameters
        ----------
        mdata: foxes.core.MData
            The mdata object
        coords: list of str
            The relevant coordinate names in the original Dataset,
            e.g. (states_coord, x_coord, y_coord). The first entry
            is interpreted as the states coordinate.
        variables: list of str
            The variables to extract from the Dataset
        
        Returns
        -------
        gpts: tuple
            Grid point data extracted from the Dataset, like
            (x, y, h) or similar
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
        # case preload
        n_states = mdata.n_states
        states_coord = coords[0]
        if self.load_mode == "preload":
            gpts = self._gpts
            weights = mdata[FV.WEIGHT] if FV.WEIGHT in mdata else None
            data = deepcopy(self._data_nostate)
            for DATA in self._data_state_keys:
                dims = mdata.dims[DATA]
                vrs = mdata[dims[-1]].tolist()
                data[dims] = (vrs, mdata[DATA].copy())

        # case lazy
        elif self.load_mode == "lazy":
            i0 = mdata.states_i0(counter=True)
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({states_coord: s}).load()
            gpts, data, weights = self._get_data(ds, variables, verbosity=0)[1:]
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
                if i0 < j0:
                    break
                else:
                    j1 = j0 + n
                    if i0 < j1:
                        a = i0 - j0
                        b = min(i1, j1) - j0
                        isel = copy(self.isel) if self.isel is not None else {}
                        isel[states_coord] = slice(a, b)

                        data.append(
                            _read_nc_file(
                                fpath,
                                coords=coords,
                                vars=vars,
                                nc_engine=config.nc_engine,
                                isel=isel,
                                sel=self.sel,
                                minimal=False,
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
            gpts, data, weights = self._get_data(data, variables, verbosity=0)[1:]
            data = {dims: (d[1], d[2]) for dims, d in data.items()}

        else:
            raise KeyError(
                f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
            )
        
        return gpts, data, weights
