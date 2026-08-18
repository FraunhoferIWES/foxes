from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Collection
from xarray import Dataset
from pathlib import Path
from typing import Any

from foxes.core import (
    Algorithm,
    FData,
    LoadedData,
    MData,
    Model,
    States,
    TData,
    VerticalProfile,
)
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
    data_source
        Either a path to a file or the data itself.
    ovars
        The output variables.
    var2col
        Mapping from variable names to data column names.
    fixed_vars
        Fixed uniform variable values, instead of reading from data.
    profdicts
        Mapping from output variable names to profile definitions.
    rpars
        Pandas file reading parameters.
    states_sel
        State subset selection.
    states_loc
        State index selection via pandas loc.
    RDICT
        Default pandas file reading parameters.


    """

    RDICT = {"index_col": 0}

    def __init__(
        self,
        data_source: str | Path | pd.DataFrame | None,
        output_vars: Collection[str],
        var2col: dict[str, str] | None = None,
        fixed_vars: dict[str, float] | None = None,
        profiles: dict[str, str | dict[str, object] | VerticalProfile] | None = None,
        read_pars: dict[str, object] | None = None,
        states_sel: slice | range | list[int] | None = None,
        states_loc: list[object] | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            Either a path to a file or the data itself.
        output_vars
            The output variables.
        var2col
            Mapping from variable names to data column names.
        fixed_vars
            Fixed uniform variable values, instead of reading from data.
        profiles
            Mapping from output variable names to profile definitions.
        read_pars
            Pandas file reading parameters.
        states_sel
            State subset selection.
        states_loc
            State index selection via pandas loc.

        """
        super().__init__()

        self.ovars = list(output_vars)
        self.rpars = {} if read_pars is None else read_pars
        self.var2col = {} if var2col is None else var2col
        self.fixed_vars = {} if fixed_vars is None else fixed_vars
        self.profdicts = {} if profiles is None else profiles
        self.states_sel = states_sel
        self.states_loc = states_loc

        if self.states_loc is not None and self.states_sel is not None:
            raise ValueError(
                f"States '{self.name}': Cannot handle both 'states_sel' and 'states_loc', please pick one"
            )

        self._N: int = 0
        self._tvars: list[str] = []
        self._profiles: dict[str, VerticalProfile] = {}
        self._data = data_source
        self.__inds: np.ndarray = np.array([], dtype=config.dtype_int)

    @property
    def data_source(self) -> pd.DataFrame | Path | str | None:
        """
        The data source

        Returns
        -------
        s
            The data source

        """
        if self.running:
            raise ValueError(
                f"States '{self.name}': Cannot access data_source while running"
            )
        return self._data

    def reset(
        self,
        algo: Algorithm | None = None,
        states_sel: slice | range | list[int] | None = None,
        states_loc: list[object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Reset the states, optionally select states

        Parameters
        ----------
        states_sel
            States subset selection
        states_loc
            State index selection via pandas loc function
        verbosity
            The verbosity level, 0 = silent

        """
        if self.initialized:
            if algo is None:
                raise KeyError(f"{self.name}: Missing algo for reset")
            elif algo.states is not self:
                raise ValueError(f"{self.name}: algo.states differs from self")
            self.finalize(algo, verbosity)
        self.states_sel = states_sel
        self.states_loc = states_loc

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        self._profiles = {}
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self._profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self._profiles[v] = d
            elif isinstance(d, dict):
                profile_type = d.pop("type")
                if not isinstance(profile_type, str):
                    raise TypeError(
                        f"States '{self.name}': Profile type for variable '{v}' must be str, got {type(profile_type).__name__}"
                    )
                self._profiles[v] = VerticalProfile.new(profile_type, **d)
            else:
                raise TypeError(
                    f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict"
                )

        return super().initialize(algo, loaded_data, force=force, verbosity=verbosity)

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return list(self._profiles.values())

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
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
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        self.VARS = self.var("vars")
        self.DATA = self.var("data")
        self.WEIGHT = self.var(FV.WEIGHT)

        if not force and self.DATA in loaded_data["data_vars"]:
            return

        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source
        else:
            data_source = self.data_source
            assert isinstance(data_source, (str, Path))
            fpath = get_input_path(data_source)
            self._data = fpath
            if not fpath.is_file():
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Reading static data '{fpath}' from context '{STATES}'"
                    )
                fpath = algo.dbook.get_file_path(STATES, fpath.name, check_raw=False)
                self._data = fpath
                if verbosity > 0:
                    print(f"Path: {fpath}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {fpath}")
            rpars = dict(self.RDICT, **self.rpars)
            data = PandasFileHelper().read_file(fpath, **rpars)

        if self.states_sel is not None:
            data = data.iloc[self.states_sel]
        elif self.states_loc is not None:
            data = data.loc[self.states_loc]
        self._N = len(data.index)
        self.__inds = data.index.to_numpy()

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        weights = None
        if col_w in data:
            weights = data[col_w].to_numpy()
        elif FV.WEIGHT in self.var2col:
            raise KeyError(
                f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {data.columns}"
            )

        tvars = set(self.ovars)
        for v in self.profdicts.keys():
            tvars.update(self._profiles[v].input_vars())
        tvars -= set(self.fixed_vars.keys())
        self._tvars = list(tvars)

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

        loaded_data["coords"][self.VARS] = np.asarray(self._tvars, dtype=str)
        loaded_data["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data.to_numpy())
        if weights is not None:
            loaded_data["data_vars"][self.WEIGHT] = ((FC.STATE,), weights)

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int
            The total number of states

        """
        return self._N

    def index(self) -> np.ndarray:
        """
        The index list

        Returns
        -------
        indices
            The index labels of states, or None for default integers

        """
        if self.running:
            raise ValueError(f"States '{self.name}': Cannot access index while running")
        return self.__inds

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
                data_source=self._data,
                inds=self.__inds,
            )
        del self._data, self.__inds

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
            self._data = data.pop("data_source")
            self.__inds = np.asarray(data.pop("inds"))

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData | None = None,
        fdata: FData | None = None,
        tdata: TData | None = None,
        *args: object,
        **parameters: object,
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
        args
            Additional positional parameters for extension compatibility
        parameters
            Additional keyword parameters for extension compatibility

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values
            (n_states, n_targets, n_tpoints)

        """
        if mdata is None or fdata is None or tdata is None:
            raise KeyError(
                f"States '{self.name}': Missing input data for calculate(), expected mdata, fdata and tdata"
            )

        super().calculate(algo, mdata, fdata, tdata)

        for i, v in enumerate(self._tvars):
            tdata[v][:] = mdata[self.DATA][:, i, None, None]

        for v, f in self.fixed_vars.items():
            tdata[v][:] = f

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


class Timeseries(StatesTable):
    """
    Timeseries states data.


    """

    RDICT: dict[str, Any] = {"index_col": 0, "parse_dates": [0]}


class TabStates(StatesTable):
    """
    States created from a single tab file


    """

    def __init__(
        self,
        data_source: str | Path | Dataset,
        output_vars: Collection[str],
        var2col: dict[str, str] | None = None,
        fixed_vars: dict[str, float] | None = None,
        profiles: dict[str, str | dict[str, object] | VerticalProfile] | None = None,
        read_pars: dict[str, object] | None = None,
        states_sel: slice | range | list[int] | None = None,
        states_loc: list[object] | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The tab file data file name, or its data
        output_vars
            The output variables
        var2col
            Mapping from variable names to data column names
        fixed_vars
            Fixed uniform variable values
        profiles
            Vertical profile definitions by variable.
        read_pars
            pandas file reading parameters
        states_sel
            States subset selection
        states_loc
            State index selection via pandas loc function
        normalize
            Normalize the tab file data

        """
        self._normalize = normalize
        self.__tab_data: Dataset | None
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

        super().__init__(
            data_source=None,
            output_vars=output_vars,
            var2col=var2col,
            fixed_vars=fixed_vars,
            profiles=profiles,
            read_pars=read_pars,
            states_sel=states_sel,
            states_loc=states_loc,
        )

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        if self.data_source is None:
            if self.__tab_data is None:
                assert self.__tab_source is not None
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
            ws: np.ndarray = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
            wd: np.ndarray = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
            ws[:] = ws0[:, None]
            wd[:] = wd0[None, :]

            wd_freq = self.__tab_data["wd_freq"].to_numpy() / 100
            weights = self.__tab_data["ws_freq"].to_numpy() * wd_freq[None, :] / 1000

            sel = weights > 0

            tab_frame = pd.DataFrame(
                index=np.arange(np.sum(sel)),
                data={
                    FV.WS: ws[sel],
                    FV.WD: wd[sel],
                    FV.WEIGHT: weights[sel],
                },
            )
            tab_frame.index.name = FC.STATE
            self._data = tab_frame

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

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
            data_stash[self.name].update(
                dict(
                    tab_source=self.__tab_source,
                    tab_data=self.__tab_data,
                )
            )
        del self.__tab_source, self.__tab_data

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
            tab_source = data.pop("tab_source")
            tab_data = data.pop("tab_data")
            if not isinstance(tab_source, (str, Path, type(None))):
                raise TypeError(
                    f"States '{self.name}': Invalid restored tab source type {type(tab_source).__name__}"
                )
            if not isinstance(tab_data, (Dataset, type(None))):
                raise TypeError(
                    f"States '{self.name}': Invalid restored tab data type {type(tab_data).__name__}"
                )
            self.__tab_source = tab_source
            self.__tab_data = tab_data
