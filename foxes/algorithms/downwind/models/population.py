import numpy as np
from xarray import Dataset, open_dataset

from foxes.config import config, get_input_path
from foxes.core import TurbineModel, States, MData, FData, TData
import foxes.constants as FC
import foxes.variables as FV


class PopulationStates(States):
    """
    States extended by a population factor.

    For each original state, n_pop states are created.
    This is useful for parameter studies, where each
    inserted state corresponds to a different value of the
    associated variable.

    Attributes
    ----------
    states: foxes.core.States
        The original states
    n_pop: int
        The population size

    :group: core

    """

    def __init__(self, states, n_pop):
        """
        Constructor.

        Parameters
        ----------
        states: foxes.core.States
            The original states
        n_pop: int
            The population size

        """
        super().__init__()
        self.states = states
        self.n_pop = n_pop

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
        self.STATE0 = self.var(FC.STATE + "0")
        self.SMAP = self.var("SMAP")

        idata = super().load_data(algo, verbosity)
        idata0 = algo.get_model_data(self.states)
        n_states0 = self.states.size()
        for cname, coord in idata0["coords"].items():
            if cname != FC.STATE:
                idata["coords"][cname] = coord
            else:
                idata["coords"][self.STATE0] = coord

        for dname, (dims0, data0) in idata0["data_vars"].items():
            hdims = tuple(
                [d if d != FC.STATE else self.STATE0 for d in np.atleast_1d(dims0)]
            )
            idata["data_vars"][dname] = (hdims, data0)
        if FV.WEIGHT not in idata["data_vars"]:
            idata["data_vars"][FV.WEIGHT] = (
                (self.STATE0,),
                np.full(n_states0, 1 / n_states0, dtype=config.dtype_double),
            )

        smap = np.zeros((self.n_pop, self.states.size()), dtype=np.int32)
        smap[:] = np.arange(self.states.size())[None, :]
        smap = smap.reshape(self.size())
        idata["data_vars"][self.SMAP] = ((FC.STATE,), smap)

        found = False
        for dname, (dims0, data0) in idata["data_vars"].items():
            if self.STATE0 in dims0:
                found = True
                break
        if not found:
            del idata["coords"][self.STATE0]

        return idata

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.states.initialized:
            self.states.initialize(algo, verbosity)
        super().initialize(algo, verbosity)

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size() * self.n_pop

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
        return self.states.output_point_vars(algo)

    def calculate(self, algo, mdata, fdata, tdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        smap = mdata[self.SMAP]

        def _map(in_data, DClass):
            if in_data is None:
                return None

            hdata = {}
            hdims = {}
            for dname, data in in_data.items():
                dms = in_data.dims[dname]
                if dname == self.SMAP or dname == self.STATE0:
                    pass
                elif dms[0] == self.STATE0:
                    hdata[dname] = data[smap]
                    hdims[dname] = tuple([FC.STATE] + list(dms)[1:])
                elif self.STATE0 in dms:
                    raise ValueError(
                        f"States '{self.name}': Found states variable not at dimension 0 for mdata entry '{dname}': {dms}"
                    )
                else:
                    hdata[dname] = data
                    hdims[dname] = dms
            return DClass(hdata, hdims, name=in_data.name + "_pop")

        hmdata = _map(mdata, MData)
        hfdata = _map(fdata, FData)
        htdata = _map(tdata, TData)
        out = self.states.calculate(algo, hmdata, hfdata, htdata)
        del hmdata, hfdata

        assert FV.WEIGHT in htdata, (
            f"Missing '{FV.WEIGHT}' in tdata results from states '{self.states.name}'"
        )
        out[FV.WEIGHT] = np.zeros(
            (htdata.n_states, htdata.n_targets, htdata.n_tpoints),
            dtype=config.dtype_double,
        )
        out[FV.WEIGHT][:] = htdata[FV.WEIGHT]

        return out


class PopulationModel(TurbineModel):
    """
    This model manages parameter studies by introducing
    a population into the states

    Attributes
    ----------
    index_coord: str
        The name of the index coordinate, labeling individuals
        within the population.
    turbine_coord: str
        The name of the turbine coordinate
    var2ncvar: dict
        Mapping from variable names to NetCDF variable names
    variables: list of str
        The variables to be set. If None, all variables
        fields from the dataset are used

    :group: core

    """

    def __init__(
        self,
        data_source,
        index_coord="index",
        turbine_coord="turbine",
        var2ncvar={},
        variables=None,
        verbosity=1,
        **kwargs,
    ):
        """
        Constructor

        Parameters
        ----------
        data_source: xarray.Dataset or str
            The population data or path to NetCDF file.
        index_coord: str
            The name of the index coordinate, labeling individuals
            within the population.
        turbine_coord: str
            The name of the turbine coordinate
        var2ncvar: dict
            Mapping from variable names to NetCDF variable names
        variables: list of str, optional
            The variables to be set. If None, all variables
            fields from the dataset are used#
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.index_coord = index_coord
        self.turbine_coord = turbine_coord
        self.var2ncvar = var2ncvar
        self.variables = variables

        # n_pop is needed very early, hence the file is loaded here
        if isinstance(data_source, Dataset):
            self._data = data_source
        else:
            fpath = get_input_path(data_source)
            if verbosity > 0:
                print(
                    f"PopulationModel '{self.name}': Loading population data from '{fpath}'"
                )
            self._data = open_dataset(fpath)
        self.__n_pop = self._data.sizes[self.index_coord]

        if variables is None:
            ncvar2var = {ncv: v for v, ncv in self.var2ncvar.items()}
            self.variables = [
                ncvar2var.get(ncv, ncv)
                for ncv in self._data.data_vars.keys()
                if self._data[ncv].dims == (self.index_coord, self.turbine_coord)
            ]
            if verbosity > 0:
                print(
                    f"PopulationModel '{self.name}': Detected variables {self.variables}"
                )
            assert len(self.variables) > 0, (
                f"PopulationModel '{self.name}': No variables found in population data with dims ({self.index_coord}, {self.turbine_coord})"
            )

    @property
    def n_pop(self):
        """
        The population size

        Returns
        -------
        pop_size: int
            The population size

        """
        return self.__n_pop

    def output_farm_vars(self, algo):
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
        return self.variables

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
        assert isinstance(algo.states, PopulationStates), (
            f"Algorithm '{algo.name}': PopulationModel '{self.name}' requires PopulationStates, found '{type(algo.states).__name__}'"
        )
        algo.init_states()

        n_states0 = algo.states.states.size()
        n_vrs = len(self.variables)
        data = np.zeros(
            (self.n_pop, n_states0, algo.n_turbines, n_vrs), dtype=config.dtype_double
        )
        for i, v in enumerate(self.variables):
            c = self.var2ncvar.get(v, v)
            assert c in self._data.data_vars, (
                f"PopulationModel '{self.name}': Missing variable '{c}' in population data, found {list(self._data.data_vars.keys())}"
            )
            data[..., i] = self._data.data_vars[c].values[:, None, :]
        data = data.reshape(algo.states.size(), algo.n_turbines, n_vrs)

        self.DATA = self.var("DATA")
        self.VARS = self.var("VARS")
        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = self.variables
        idata["data_vars"][self.DATA] = ((FC.STATE, FC.TURBINE, self.VARS), data)

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
        data_stash[self.name] = dict(data=self._data)
        del self._data

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
        self._data = data.pop("data")

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        self.ensure_output_vars(algo, fdata)

        data = mdata[self.DATA][st_sel]
        for i, v in enumerate(self.variables):
            fdata[v][st_sel] = data[..., i]

        return {v: fdata[v] for v in self.variables}

    def farm2pop_results(self, algo, farm_results):
        """
        Convert farm results to population results

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        farm_results: xarray.Dataset
            The farm results

        Returns
        -------
        pop_results: xarray.Dataset
            The population farm results

        """
        assert isinstance(algo.states, PopulationStates), (
            f"Algorithm '{algo.name}': PopulationModel '{self.name}' requires PopulationStates, found '{type(algo.states).__name__}'"
        )

        n_states0 = algo.states.states.size()
        inds0 = algo.states.states.index()
        coords = {FC.STATE: inds0} if inds0 is not None else {}
        coords.update(
            {c: d.values for c, d in farm_results.coords.items() if c != FC.STATE}
        )

        data = {}
        for dname, d in farm_results.data_vars.items():
            if d.dims[0] == FC.STATE:
                data[dname] = (
                    (self.index_coord,) + d.dims,
                    d.values.reshape((self.n_pop, n_states0) + d.shape[1:]),
                )
            else:
                data[dname] = (d.dims, d.values)

        return Dataset(data, coords=coords)
