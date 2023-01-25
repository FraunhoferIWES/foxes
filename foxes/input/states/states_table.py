import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import States, VerticalProfile
from foxes.utils import PandasFileHelper
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC


class StatesTable(States):
    """
    States from a `pandas.DataFrame` or a pandas readable file.

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        Either path to a file or data
    output_vars : list of str
        The output variables
    var2col : dict
        Mapping from variable names to data column names
    fixed_vars : dict
        Fixed uniform variable values, instead of
        reading from data
    profiles : dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    pd_read_pars : dict
        pandas file reading parameters
    states_sel : slice or range or list of int, optional
        States subset selection
    states_loc : list, optional
        State index selection via pandas loc function

    Attributes
    ----------
    data_source : str or pandas.DataFrame
        Either path to a file or data
    ovars : list of str
        The output variables
    var2col : dict
        Mapping from variable names to data column names
    fixed_vars : dict
        Fixed uniform variable values, instead of
        reading from data
    profdicts : dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    rpars : dict
        pandas file reading parameters
    states_sel : slice or range or list of int
        States subset selection
    states_loc : list
        State index selection via pandas loc function
    RDICT : dict
        Default pandas file reading parameters

    """

    RDICT = {"index_col": 0}

    def __init__(
        self,
        data_source,
        output_vars,
        var2col={},
        fixed_vars={},
        profiles={},
        pd_read_pars={},
        states_sel=None,
        states_loc=None,
    ):
        super().__init__()

        self.data_source = data_source
        self.ovars = output_vars
        self.rpars = pd_read_pars
        self.var2col = var2col
        self.fixed_vars = fixed_vars
        self.profdicts = profiles
        self.states_sel = states_sel
        self.states_loc = states_loc

        if self.states_loc is not None and self.states_sel is not None:
            raise ValueError(f"States '{self.name}': Cannot handle both 'states_sel' and 'states_loc', please pick one")

        self._weights = None
        self._N = None
        self._tvars = None

    def reset(self, algo=None, states_sel=None, states_loc=None, verbosity=0):
        """
        Reset the states, optionally select states

        Parameters
        ----------
        states_sel : slice or range or list of int, optional
            States subset selection
        states_loc : list, optional
            State index selection via pandas loc function
        verbosity : int
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

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.VARS = self.var("vars")
        self.DATA = self.var("data")

        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source
            isorg = True
        else:
            if not Path(self.data_source).is_file():
                if verbosity:
                    print(
                        f"States '{self.name}': Reading static data '{self.data_source}' from context '{STATES}'"
                    )
                self.data_source = algo.dbook.get_file_path(
                    STATES, self.data_source, check_raw=False
                )
                if verbosity:
                    print(f"Path: {self.data_source}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {self.data_source}")
            rpars = dict(self.RDICT, **self.rpars)
            data = PandasFileHelper().read_file(self.data_source, **rpars)
            isorg = False

        self._profiles = {}
        self._tvars = set(self.ovars)
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
            self._tvars.update(self._profiles[v].input_vars())
        self._tvars -= set(self.fixed_vars.keys())
        self._tvars = list(self._tvars)

        if self.states_sel is not None:
            data = data.iloc[self.states_sel]
        elif self.states_loc is not None:
            data = data.loc[self.states_loc]
        self._N = len(data.index)

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        self._weights = np.zeros((self._N, algo.n_turbines), dtype=FC.DTYPE)
        if col_w in data:
            self._weights[:] = data[col_w].to_numpy()[:, None]
        elif FV.WEIGHT in self.var2col:
            raise KeyError(
                f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {data.columns}"
            )
        else:
            self._weights[:] = 1.0 / self._N
            if isorg:
                data = data.copy()
            data[col_w] = self._weights[:, 0]

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

        idata = super().initialize(algo, verbosity)
        self._update_idata(algo, idata)
        idata["coords"][self.VARS] = self._tvars
        idata["data_vars"][self.DATA] = ((FV.STATE, self.VARS), data.to_numpy())

        algo.update_idata(list(self._profiles.values()),
            idata=idata, verbosity=verbosity)
            
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
            The weights, shape: (n_states, n_turbines)

        """
        return self._weights

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        pdata : foxes.core.Data
            The point data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        z = pdata[FV.POINTS][:, :, 2]

        for i, v in enumerate(self._tvars):
            pdata[v][:] = mdata[self.DATA][:, i, None]

        for v, f in self.fixed_vars.items():
            pdata[v] = np.full((pdata.n_states, pdata.n_points), f, dtype=FC.DTYPE)

        for v, p in self._profiles.items():
            pres = p.calculate(pdata, z)
            pdata[v] = pres

        return {v: pdata[v] for v in self.output_point_vars(algo)}

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        self._weights = None
        self._N = None
        self._tvars = None

        super().finalize(algo, verbosity)


class Timeseries(StatesTable):
    """
    Timeseries states data.
    """

    RDICT = {"index_col": 0, "parse_dates": [0]}
