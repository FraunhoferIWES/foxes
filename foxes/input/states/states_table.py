import numpy as np
import pandas as pd
from xarray import Dataset
from pathlib import Path

from foxes.core import States, VerticalProfile
from foxes.utils import PandasFileHelper, read_tab_file
from foxes.data import STATES
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC


class StatesTable(States):
    """
    States from a `pandas.DataFrame` or a pandas readable file.

    Attributes
    ----------
    data_source: str or pandas.DataFrame
        Either path to a file or data
    ovars: list of str
        The output variables
    var2col: dict
        Mapping from variable names to data column names
    fixed_vars: dict
        Fixed uniform variable values, instead of
        reading from data
    profdicts: dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    rpars: dict
        pandas file reading parameters
    states_sel: slice or range or list of int
        States subset selection
    states_loc: list
        State index selection via pandas loc function
    RDICT: dict
        Default pandas file reading parameters

    :group: input.states

    """

    RDICT = {"index_col": 0}

    def __init__(
        self,
        data_source,
        output_vars,
        var2col={},
        fixed_vars={},
        profiles={},
        read_pars={},
        states_sel=None,
        states_loc=None,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            Either path to a file or data
        output_vars: list of str
            The output variables
        var2col: dict
            Mapping from variable names to data column names
        fixed_vars: dict
            Fixed uniform variable values, instead of
            reading from data
        profiles: dict
            Key: output variable name str, Value: str or dict
            or `foxes.core.VerticalProfile`
        read_pars: dict
            pandas file reading parameters
        states_sel: slice or range or list of int, optional
            States subset selection
        states_loc: list, optional
            State index selection via pandas loc function

        """
        super().__init__()

        self.ovars = list(output_vars)
        self.rpars = read_pars
        self.var2col = var2col
        self.fixed_vars = fixed_vars
        self.profdicts = profiles
        self.states_sel = states_sel
        self.states_loc = states_loc

        if self.states_loc is not None and self.states_sel is not None:
            raise ValueError(
                f"States '{self.name}': Cannot handle both 'states_sel' and 'states_loc', please pick one"
            )

        self._N = None
        self._tvars = None
        self._profiles = None
        self._data_source = data_source

    @property
    def data_source(self):
        """
        The data source

        Returns
        -------
        s: object
            The data source

        """
        if self.running:
            raise ValueError(
                f"States '{self.name}': Cannot access data_source while running"
            )
        return self._data_source

    def reset(self, algo=None, states_sel=None, states_loc=None, verbosity=0):
        """
        Reset the states, optionally select states

        Parameters
        ----------
        states_sel: slice or range or list of int, optional
            States subset selection
        states_loc: list, optional
            State index selection via pandas loc function
        verbosity: int
            The verbosity level, 0 = silent

        """
        if self.initialized:
            if algo is None:
                raise KeyError(f"{self.name}: Missing algo for reset")
            elif algo.states is not self:
                raise ValueError(f"{self.states}: algo.states differs from self")
            self.finalize(algo, verbosity)
        self.states_sel = states_sel
        self.states_loc = states_loc

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
        self._profiles = {}
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self._profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self._profiles[v] = d
            elif isinstance(d, dict):
                t = d.pop("type")
                self._profiles[v] = VerticalProfile.new(t, **d)
            else:
                raise TypeError(
                    f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict"
                )

        super().initialize(algo, verbosity)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of str
            Names of all sub models

        """
        return list(self._profiles.values())

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
        self.VARS = self.var("vars")
        self.DATA = self.var("data")
        self.WEIGHT = self.var(FV.WEIGHT)

        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source
        else:
            self._data_source = get_input_path(self.data_source)
            if not self.data_source.is_file():
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Reading static data '{self.data_source}' from context '{STATES}'"
                    )
                self._data_source = algo.dbook.get_file_path(
                    STATES, self.data_source.name, check_raw=False
                )
                if verbosity > 0:
                    print(f"Path: {self.data_source}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {self.data_source}")
            rpars = dict(self.RDICT, **self.rpars)
            data = PandasFileHelper().read_file(self.data_source, **rpars)

        if self.states_sel is not None:
            data = data.iloc[self.states_sel]
        elif self.states_loc is not None:
            data = data.loc[self.states_loc]
        self._N = len(data.index)
        self.__inds = data.index.to_numpy()

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        weights = None
        if col_w in data:
            weights = data[col_w].to_numpy()
        elif FV.WEIGHT in self.var2col:
            raise KeyError(
                f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {data.columns}"
            )

        self._tvars = set(self.ovars)
        for v in self.profdicts.keys():
            self._tvars.update(self._profiles[v].input_vars())
        self._tvars -= set(self.fixed_vars.keys())
        self._tvars = list(self._tvars)

        tcols = []
        for v in self._tvars:
            c = self.var2col.get(v, v)
            if c in data.columns:
                tcols.append(c)
            elif v not in self._profiles.keys():
                raise KeyError(
                    f"States '{self.name}': Missing variable '{c}' in states table columns, profiles or fixed vars"
                )
        data = data[tcols]

        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = self._tvars
        idata["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data.to_numpy())
        if weights is not None:
            idata["data_vars"][self.WEIGHT] = (FC.STATE, weights)

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
        indices: array_like
            The index labels of states, or None for default integers

        """
        if self.running:
            raise ValueError(f"States '{self.name}': Cannot access index while running")
        return self.__inds

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
            data_source=self._data_source,
            inds=self.__inds,
        )
        del self._data_source, self.__inds

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
        self._data_source = data.pop("data_source")
        self.__inds = data.pop("inds")

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
        for i, v in enumerate(self._tvars):
            if v in tdata:
                tdata[v][:] = mdata[self.DATA][:, i, None, None]
            else:
                tdata[v] = np.zeros(
                    (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                    dtype=config.dtype_double,
                )
                tdata[v][:] = mdata[self.DATA][:, i, None, None]
                tdata.dims[v] = (FC.STATE, FC.TARGET, FC.TPOINT)

        for v, f in self.fixed_vars.items():
            tdata[v] = np.full(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                f,
                dtype=config.dtype_double,
            )

        z = tdata[FC.TARGETS][..., 2]
        for v, p in self._profiles.items():
            tdata[v] = p.calculate(tdata, z)

        if self.WEIGHT in mdata:
            tdata[FV.WEIGHT] = mdata[self.WEIGHT][:, None, None]
        else:
            tdata[FV.WEIGHT] = np.full(
                (mdata.n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
            )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: tdata[v] for v in self.output_point_vars(algo)}

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level

        """
        self._N = None
        self._tvars = None

        super().finalize(algo, verbosity)


class Timeseries(StatesTable):
    """
    Timeseries states data.

    :group: input.states

    """

    RDICT = {"index_col": 0, "parse_dates": [0]}


class TabStates(StatesTable):
    """
    States created from a single tab file

    :group: input.states

    """

    def __init__(self, data_source, *args, normalize=True, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or xarray.Dataset
            The tab file data file name, or its data
        args: tuple, optional
            Additional parameters for StatesTable
        normalize: bool
            Normalize the tab file data
        kwargs: dict, optional
            Additional parameters for StatesTable

        """
        self._normalize = normalize
        if isinstance(data_source, Dataset):
            self.__tab_source = None
            self.__tab_data = data_source
        elif isinstance(data_source, (str, Path)):
            self.__tab_source = data_source
            self.__tab_data = None
        else:
            raise TypeError(
                f"Expecting str, Path or xarray.Dataset as data_source, got {type(data_source)}"
            )

        super().__init__(data_source=None, *args, **kwargs)

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
        if self.data_source is None:
            if self.__tab_data is None:
                self.__tab_source = get_input_path(self.__tab_source)
                if not self.__tab_source.is_file():
                    if verbosity > 0:
                        print(
                            f"States '{self.name}': Reading static data '{self.__tab_source}' from context '{STATES}'"
                        )
                    self.__tab_source = algo.dbook.get_file_path(
                        STATES, self.__tab_source.name, check_raw=False
                    )
                    if verbosity > 0:
                        print(f"Path: {self.__tab_source}")
                elif verbosity:
                    print(f"States '{self.name}': Reading file {self.__tab_source}")
                self.__tab_data = read_tab_file(self.__tab_source, self._normalize)

            a = self.__tab_data.attrs["factor_ws"]
            b = self.__tab_data.attrs["shift_wd"]
            if b != 0.0:
                raise ValueError(
                    f"{self.name}: shift_wd = {b} is not supported, expecting zero"
                )

            wd0 = self.__tab_data["wd"].to_numpy()
            ws0 = a * np.append(
                np.array([0], dtype=config.dtype_double),
                self.__tab_data["ws"].to_numpy(),
            )
            ws0 = 0.5 * (ws0[:-1] + ws0[1:])

            n_ws = self.__tab_data.sizes["ws"]
            n_wd = self.__tab_data.sizes["wd"]
            ws = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
            wd = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
            ws[:] = ws0[:, None]
            wd[:] = wd0[None, :]

            wd_freq = self.__tab_data["wd_freq"].to_numpy() / 100
            weights = self.__tab_data["ws_freq"].to_numpy() * wd_freq[None, :] / 1000

            sel = weights > 0

            self._data_source = pd.DataFrame(
                index=np.arange(np.sum(sel)),
                data={
                    FV.WS: ws[sel],
                    FV.WD: wd[sel],
                    FV.WEIGHT: weights[sel],
                },
            )
            self._data_source.index.name = FC.STATE

        return super().load_data(algo, verbosity)

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

        data_stash[self.name].update(
            dict(
                tab_source=self.__tab_source,
                tab_data=self.__tab_data,
            )
        )
        del self.__tab_source, self.__tab_data

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
        self.__tab_source = data.pop("tab_source")
        self.__tab_data = data.pop("tab_data")
