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

    def __init__(self, states, n_pop, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        states: foxes.core.States
            The original states
        n_pop: int
            The population size
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(load_mode=states.load_mode, **kwargs)
        self.states = states
        self.n_pop = n_pop

    def sub_models(self):
        """
        List of all sub-models.

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.states]

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

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size() * self.n_pop

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        """
        Load and/or create all data required for model calculations.

        The function adds to loaded_data.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """

        # load sub model and memorize new entries:
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        # prepare:
        self.STATE0 = self.var(FC.STATE + "0")
        self.SMAP = self.var("smap")
        n_states0 = self.states.size()
        coords = loaded_data["coords"]
        data_vars = loaded_data["data_vars"]

        # load only once:
        if self.SMAP in data_vars:
            return

        # reset states dimension:
        coords[self.STATE0] = self.states.index()
        need_state0 = False
        dkeys = list(data_vars.keys())
        for dname in dkeys:
            if FC.STATE in data_vars[dname][0]:
                dims, data = data_vars.pop(dname)
                dims = tuple([self.STATE0 if d == FC.STATE else d for d in dims])
                data_vars[dname] = (dims, data)
                need_state0 = True

        # make sure that the weight variable is present:
        if FV.WEIGHT not in data_vars:
            data_vars[FV.WEIGHT] = (
                (self.STATE0,),
                np.full(n_states0, 1 / n_states0, dtype=config.dtype_double),
            )
            need_state0 = True

        # create mapping from new states to original states:
        smap = np.zeros((self.states.size(), self.n_pop), dtype=config.dtype_int)
        smap[:] = np.arange(self.states.size())[:, None]
        smap = smap.reshape(self.size())
        data_vars[self.SMAP] = ((FC.STATE,), smap)

        # remove state0 from coords if not needed:
        if self.STATE0 in coords and not need_state0:
            coords.pop(self.STATE0)

    def load_chunk_data(self, algo, mdata, fdata, tdata):
        """
        Load chunk data according to load mode.

        This function adds data to mdata.

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

        """
        if self.load_mode == "preload":
            return

        # prepare mdata:
        smap = mdata[self.SMAP]
        i0 = np.min(smap)
        i1 = np.max(smap) + 1
        if self.STATE0 in mdata:
            data = {FC.STATE: mdata[self.STATE0][i0:i1]}
        else:
            data = {FC.STATE: np.arange(i0, i1, dtype=config.dtype_int)}
        dims = {FC.STATE: (FC.STATE,)}
        for dname, data in mdata.items():
            dms = mdata.dims[dname]
            if dname == self.SMAP or dname == self.STATE0:
                pass
            elif dms[0] == self.STATE0:
                data[dname] = data[smap]
                dims[dname] = tuple([FC.STATE] + list(dms)[1:])
            elif self.STATE0 in dms:
                raise ValueError(
                    f"States '{self.name}': Expecting {self.STATE0} at position 0 for {dname}, got {dms}"
                )
            else:
                data[dname] = data
                dims[dname] = dms
        sub_mdata = MData(
            data=data,
            dims=dims,
            chunki_states=mdata.chunki_states,
            chunki_points=mdata.chunki_points,
            n_chunks_states=mdata.n_chunks_states,
            n_chunks_points=mdata.n_chunks_points,
            extra_data=mdata.extra_data,
            name=f"{mdata.name}_sub",
        )

        # load sub model chunk data:
        keys0 = set(mdata.keys())
        super().load_chunk_data(algo, sub_mdata, fdata=None, tdata=None)
        new_keys = set(mdata.keys()) - keys0

        # add new data to mdata:
        if FC.STATE in new_keys:
            mdata[self.STATE0] = mdata.pop(FC.STATE)
            mdata.dims[self.STATE0] = mdata.dims.pop(FC.STATE)
            new_keys.remove(FC.STATE)
        else:
            mdata[self.STATE0] = sub_mdata[FC.STATE]
            mdata.dims[self.STATE0] = (self.STATE0,)
        for dname in new_keys:
            data = sub_mdata[dname]
            dms = sub_mdata.dims[dname]
            if dms[0] == FC.STATE:
                mdata[dname] = data
                mdata.dims[dname] = tuple([self.STATE0] + list(dms)[1:])
            elif FC.STATE in dms:
                raise ValueError(
                    f"States '{self.name}': Expecting {FC.STATE} at position 0 for {dname}, got {dms} from states '{self.states.name}'"
                )
            else:
                mdata[dname] = data
                mdata.dims[dname] = dms

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
        super().calculate(algo, mdata, fdata, tdata)

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
            return DClass.from_data(
                in_data, data=hdata, dims=hdims, name=in_data.name + "_pop"
            )

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

        # ensure that all states have full dimensions:
        for v in out.keys():
            if out[v].shape[0] == 1 and htdata.n_states > 1:
                tmp = out[v]
                out[v] = np.zeros((htdata.n_states,) + tmp.shape[1:], dtype=tmp.dtype)
                out[v][:] = tmp
                del tmp

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

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        """
        Load and/or create all data required for model calculations.

        The function adds to loaded_data.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        self.DATA = self.var("DATA")
        self.VARS = self.var("VARS")
        if self.DATA in loaded_data["data_vars"]:
            return

        assert isinstance(algo.states, PopulationStates), (
            f"Algorithm '{algo.name}': PopulationModel '{self.name}' requires PopulationStates, found '{type(algo.states).__name__}'"
        )
        algo.init_states()

        self.n_states0 = algo.states.states.size()
        self._inds0 = algo.states.states.index()
        n_vrs = len(self.variables)
        data = np.zeros(
            (self.n_pop, self.n_states0, algo.n_turbines, n_vrs),
            dtype=config.dtype_double,
        )
        for i, v in enumerate(self.variables):
            c = self.var2ncvar.get(v, v)
            assert c in self._data.data_vars, (
                f"PopulationModel '{self.name}': Missing variable '{c}' in population data, found {list(self._data.data_vars.keys())}"
            )
            data[..., i] = self._data.data_vars[c].values[:, None, :]
        data = data.reshape(algo.states.size(), algo.n_turbines, n_vrs)

        loaded_data["coords"][self.VARS] = self.variables
        loaded_data["data_vars"][self.DATA] = ((FC.STATE, FC.TURBINE, self.VARS), data)

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
        data_stash[self.name] = dict(data=self._data, inds0=self._inds0)
        del self._data, self._inds0

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
        self._inds0 = data.pop("inds0")

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

        coords = {FC.STATE: self._inds0} if self._inds0 is not None else {}
        coords.update(
            {c: d.values for c, d in farm_results.coords.items() if c != FC.STATE}
        )

        data = {}
        for dname, d in farm_results.data_vars.items():
            if d.dims[0] == FC.STATE:
                data[dname] = (
                    (FC.POP,) + d.dims,
                    np.swapaxes(
                        d.values.reshape((self.n_states0, self.n_pop) + d.shape[1:]),
                        0,
                        1,
                    ),
                )
            else:
                data[dname] = (d.dims, d.values)

        return Dataset(data, coords=coords)
