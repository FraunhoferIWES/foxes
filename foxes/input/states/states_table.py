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
    var2col : dict, optional
        Mapping from variable names to data column names
    fixed_vars : dict, optional
        Fixed uniform variable values, instead of
        reading from data
    profiles : dict, optional
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    pd_read_pars : dict, optional
        pandas file reading parameters

    Attributes
    ----------
    oars : list of str
        The output variables
    var2col : dict, optional
        Mapping from variable names to data column names
    fixed_vars : dict, optional
        Fixed uniform variable values, instead of
        reading from data
    profdicts : dict, optional
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    pd_read_pars : dict, optional
        pandas file reading parameters
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
    ):
        super().__init__()

        self._data = data_source
        self.ovars = output_vars
        self.rpars = pd_read_pars
        self.var2col = var2col
        self.fixed_vars = fixed_vars
        self.profdicts = profiles

    def initialize(self, algo, states_sel=None, verbosity=1):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        states_sel : slice or range or list of int, optional
            States subset selection
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        if not isinstance(self._data, pd.DataFrame):

            if not Path(self._data).is_file():
                if verbosity:
                    print(
                        f"States '{self.name}': Reading static data '{self._data}' from context '{STATES}'"
                    )
                self._data = algo.dbook.get_file_path(
                    STATES, self._data, check_raw=False
                )
                if verbosity:
                    print(f"Path: {self._data}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {self._data}")
            rpars = dict(self.RDICT, **self.rpars)
            self._data = PandasFileHelper().read_file(self._data, **rpars)

        if states_sel is not None:
            self._data = self._data.iloc[states_sel]
        self._N = len(self._data.index)

        self.profiles = {}
        self.tvars = set(self.ovars)
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self.profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self.profiles[v] = d
            elif isinstance(d, dict):
                t = d.pop("type")
                self.profiles[v] = VerticalProfile.new(t, **d)
            else:
                raise TypeError(
                    f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict"
                )
            self.tvars.update(self.profiles[v].input_vars())
        self.tvars -= set(self.fixed_vars.keys())
        self.tvars = list(self.tvars)

        for p in self.profiles.values():
            if not p.initialized:
                p.initialize(algo)

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.VARS = self.var("vars")
        self.DATA = self.var("data")

        col_w = self.var2col.get(FV.WEIGHT, FV.WEIGHT)
        self._weights = np.zeros((self._N, algo.n_turbines), dtype=FC.DTYPE)
        if col_w in self._data:
            self._weights[:] = self._data[col_w].to_numpy()[:, None]
        elif FV.WEIGHT in self.var2col:
            raise KeyError(
                f"Weight variable '{col_w}' defined in var2col, but not found in states table columns {self._data.columns}"
            )
        else:
            self._weights[:] = 1.0 / self._N

        tcols = []
        for v in self.tvars:
            c = self.var2col.get(v, v)
            if c in self._data.columns:
                tcols.append(c)
            elif v not in self.profiles.keys():
                raise KeyError(
                    f"States '{self.name}': Missing variable '{c}' in states table columns, profiles or fixed vars"
                )
        self._data = self._data[tcols]

        idata = super().model_input_data(algo)

        if self._data.index.name is not None:
            idata["coords"][FV.STATE] = self._data.index.to_numpy()

        idata["coords"][self.VARS] = self.tvars
        idata["data_vars"][self.DATA] = ((FV.STATE, self.VARS), self._data.to_numpy())

        del self._data

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
            The weights, shape: (n_states,)

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

        for i, v in enumerate(self.tvars):
            pdata[v][:] = mdata[self.DATA][:, i, None]

        for v, f in self.fixed_vars.items():
            pdata[v] = np.full((pdata.n_states, pdata.n_points), f, dtype=FC.DTYPE)

        for v, p in self.profiles.items():
            pres = p.calculate(pdata, z)
            pdata[v] = pres

        return {v: pdata[v] for v in self.output_point_vars(algo)}
