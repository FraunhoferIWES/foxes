import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from foxes.core import States
from foxes.utils import PandasFileHelper
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC


class MultiHeightStates(States):
    """
    States with multiple heights data per entry.

    The input data is taken from a csv file or
    pandas data frame with columns. The format
    of the data columns is as in the following
    example for wind speed at heights 50, 60, 100 m:

    WS-50, WS-60, WS-100, ...

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        Either path to a file or data
    output_vars : list of str
        The output variables
    heights : list of float
        The heights at which to search data
    var2col : dict, optional
        Mapping from variable names to data column names
    fixed_vars : dict, optional
        Fixed uniform variable values, instead of
        reading from data
    pd_read_pars : dict, optional
        pandas file reading parameters
    ipars : dict, optional
        Parameters for scipy.interpolate.interp1d

    Attributes
    ----------
    ovars : list of str
        The output variables
    heights : list of float
        The heights at which to search data
    var2col : dict, optional
        Mapping from variable names to data column names
    fixed_vars : dict, optional
        Fixed uniform variable values, instead of
        reading from data
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
        heights,
        var2col={},
        fixed_vars={},
        pd_read_pars={},
        ipars={},
    ):
        super().__init__()

        self.ovars = output_vars
        self.heights = np.array(heights, dtype=FC.DTYPE)
        self.rpars = pd_read_pars
        self.var2col = var2col
        self.fixed_vars = fixed_vars
        self.ipars = ipars

        self._data0 = data_source
        self._data = None
        self._cmap = None
        self._solo = None
        self._weights = None
        self._N = None

    def _find_cols(self, v, cols):
        """
        Helper function for searching height columns
        """
        c0 = self.var2col.get(v, v)
        if v in self.fixed_vars:
            return []
        elif c0 in cols:
            return [c0]
        else:
            cls = []
            for h in self.heights:
                hh = int(h) if int(h) == h else h
                c = f"{c0}-{hh}"
                oc = self.var2col.get(c, c)
                if oc in cols:
                    cls.append(oc)
                else:
                    raise KeyError(
                        f"Missing: '{v}' in fixed_vars, or '{c0}' or '{oc}' in columns. Maybe make use of var2col?"
                    )
            return cls

    def initialize(self, algo, states_sel=None, states_loc=None, verbosity=1):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        states_sel : slice or range or list of int, optional
            States subset selection
        states_loc : list, optional
            State index selection via pandas loc function
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        if not isinstance(self._data0, pd.DataFrame):
            if not Path(self._data0).is_file():
                if verbosity:
                    print(
                        f"States '{self.name}': Reading static data '{self._data0}' from context '{STATES}'"
                    )
                self._data0 = algo.dbook.get_file_path(
                    STATES, self._data0, check_raw=False
                )
                if verbosity:
                    print(f"Path: {self._data0}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {self._data0}")
            rpars = dict(self.RDICT, **self.rpars)
            self._data0 = PandasFileHelper().read_file(self._data0, **rpars)

        if states_sel is not None:
            self._data = self._data0.iloc[states_sel]
        elif states_loc is not None:
            self._data = self._data0.loc[states_loc]
        else:
            self._data = self._data0
        self._N = len(self._data.index)

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

        cols = []
        self._cmap = {}
        self._solo = {}
        for v in self.ovars:
            vcols = self._find_cols(v, self._data.columns)
            if len(vcols) == 1:
                self._solo[v] = self._data[vcols[0]].to_numpy()
            elif len(vcols) > 1:
                self._cmap[v] = (len(cols), len(cols) + len(vcols))
                cols += vcols
        self._data = self._data[cols]

        self.H = self.var(FV.H)
        self.VARS = self.var("vars")
        self.DATA = self.var("data")

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
        idata = super().model_input_data(algo)

        if self._data.index.name is not None:
            idata["coords"][FV.STATE] = self._data.index.to_numpy()
        idata["coords"][self.H] = self.heights
        idata["coords"][self.VARS] = list(self._cmap.keys())

        self._cmap = None

        n_hts = len(self.heights)
        n_vrs = int(len(self._data.columns) / n_hts)
        dims = (FV.STATE, self.VARS, self.H)
        idata["data_vars"][self.DATA] = (
            dims,
            self._data.to_numpy().reshape(self._N, n_vrs, n_hts),
        )

        self._data = None

        for v, d in self._solo.items():
            idata["data_vars"][self.var(v)] = ((FV.STATE,), d)

        self._solo = list(self._solo.keys())

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
        h = mdata[self.H]
        z = pdata[FV.POINTS][:, :, 2]
        n_h = len(h)

        coeffs = np.zeros((n_h, n_h), dtype=FC.DTYPE)
        np.fill_diagonal(coeffs, 1.0)
        ipars = dict(assume_sorted=True, bounds_error=True)
        ipars.update(self.ipars)
        intp = interp1d(h, coeffs, axis=0, **ipars)
        ires = intp(z)
        del coeffs, intp

        ires = np.einsum("svh,sph->svp", mdata[self.DATA], ires)

        results = {}
        vrs = list(mdata[self.VARS])
        for v in self.ovars:
            results[v] = pdata[v]
            if v in self.fixed_vars:
                results[v][:] = self.fixed_vars[v]
            elif v in self._solo:
                results[v][:] = mdata[self.var(v)][:, None]
            else:
                results[v] = ires[:, vrs.index(v)]

        return results

    def finalize(self, algo, results, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level

        """
        self._data = None
        self._cmap = None
        self._solo = None
        self._weights = None
        self._N = None

        super().finalize(algo, results, clear_mem, verbosity)


class MultiHeightTimeseries(MultiHeightStates):
    """
    Multi-height timeseries states data.
    """

    RDICT = {"index_col": 0, "parse_dates": [0]}
