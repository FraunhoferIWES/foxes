import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from foxes.core import States
from foxes.utils import PandasFileHelper
from foxes.data import STATES
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils import wd2uv, uv2wd


class MultiHeightStates(States):
    """
    States with multiple heights data per entry.

    The input data is taken from a csv file or
    pandas data frame with columns. The format
    of the data columns is as in the following
    example for wind speed at heights 50, 60, 100 m:

    WS-50, WS-60, WS-100, ...

    Attributes
    ----------
    data_source: str or pandas.DataFrame
        Either path to a file or data
    ovars: list of str
        The output variables
    heights: list of float
        The heights at which to search data
    var2col: dict, optional
        Mapping from variable names to data column names
    fixed_vars: dict, optional
        Fixed uniform variable values, instead of
        reading from data
    pd_read_pars: dict, optional
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
        heights,
        var2col={},
        fixed_vars={},
        pd_read_pars={},
        states_sel=None,
        states_loc=None,
        ipars={},
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            Either path to a file or data
        output_vars: list of str
            The output variables
        heights: list of float
            The heights at which to search data
        var2col: dict, optional
            Mapping from variable names to data column names
        fixed_vars: dict, optional
            Fixed uniform variable values, instead of
            reading from data
        pd_read_pars: dict, optional
            pandas file reading parameters
        states_sel: slice or range or list of int, optional
            States subset selection
        states_loc: list, optional
            State index selection via pandas loc function
        ipars: dict, optional
            Parameters for scipy.interpolate.interp1d

        """
        super().__init__()

        self.data_source = data_source
        self.ovars = output_vars
        self.heights = np.array(heights, dtype=FC.DTYPE)
        self.rpars = pd_read_pars
        self.var2col = var2col
        self.fixed_vars = fixed_vars
        self.ipars = ipars
        self.states_sel = states_sel
        self.states_loc = states_loc

        self._cmap = None
        self._solo = None
        self._weights = None
        self._N = None

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
        if not isinstance(self.data_source, pd.DataFrame):
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
        else:
            isorg = True
            data = self.data_source

        if self.states_sel is not None:
            data = data.iloc[self.states_sel]
        elif self.states_loc is not None:
            data = data.loc[self.states_loc]

        self._N = len(data.index)
        self._inds = data.index.to_numpy()

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

        cols = []
        self._cmap = {}
        self._solo = {}
        for v in self.ovars:
            vcols = self._find_cols(v, data.columns)
            if len(vcols) == 1:
                self._solo[v] = data[vcols[0]].to_numpy()
            elif len(vcols) > 1:
                self._cmap[v] = (len(cols), len(cols) + len(vcols))
                cols += vcols
        data = data[cols]

        self.H = self.var(FV.H)
        self.VARS = self.var("vars")
        self.DATA = self.var("data")

        idata = super().load_data(algo, verbosity)

        idata["coords"][self.H] = self.heights
        idata["coords"][self.VARS] = list(self._cmap.keys())

        n_hts = len(self.heights)
        n_vrs = int(len(data.columns) / n_hts)
        dims = (FC.STATE, self.VARS, self.H)
        idata["data_vars"][self.DATA] = (
            dims,
            data.to_numpy().reshape(self._N, n_vrs, n_hts),
        )

        for v, d in self._solo.items():
            idata["data_vars"][self.var(v)] = ((FC.STATE,), d)

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
        return self._inds

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

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights: numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        return self._weights

    def calculate(self, algo, mdata, fdata, tdata):
        """ "
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
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        h = mdata[self.H]
        z = tdata[FC.TARGETS][..., 2].reshape(n_states, n_targets * n_tpoints)
        n_h = len(h)
        vrs = list(mdata[self.VARS])
        n_vars = len(vrs)

        coeffs = np.zeros((n_h, n_h), dtype=FC.DTYPE)
        np.fill_diagonal(coeffs, 1.0)
        ipars = dict(assume_sorted=True, bounds_error=True)
        ipars.update(self.ipars)
        intp = interp1d(h, coeffs, axis=0, **ipars)
        ires = intp(z)
        del coeffs, intp

        has_wd = FV.WD in vrs
        if has_wd:
            i_wd = vrs.index(FV.WD)
            if FV.WS in vrs:
                i_ws = vrs.index(FV.WS)
                uvh = wd2uv(
                    mdata[self.DATA][:, i_wd], mdata[self.DATA][:, i_ws], axis=-1
                )
            elif FV.WS in self.fixed_vars:
                uvh = wd2uv(mdata[self.DATA][:, i_wd], self.fixed_vars[FV.WS], axis=-1)
            elif self.var(FV.WS) in mdata:
                uvh = wd2uv(
                    mdata[self.DATA][:, i_wd], mdata[self.var(FV.WS)][:, None], axis=-1
                )
            else:
                raise KeyError(
                    f"States '{self.name}': Found variable '{FV.WD}', but missing variable '{FV.WS}'"
                )
            uv = np.einsum("shd,sph->spd", uvh, ires).reshape(
                n_states, n_targets, n_tpoints, 2
            )
            del uvh

        ires = np.einsum("svh,sph->vsp", mdata[self.DATA], ires).reshape(
            n_vars, n_states, n_targets, n_tpoints
        )

        results = {}
        for v in self.ovars:
            if has_wd and v == FV.WD:
                results[v] = uv2wd(uv, axis=-1)
            elif has_wd and v == FV.WS:
                results[v] = np.linalg.norm(uv, axis=-1)
            elif v in self.fixed_vars:
                results[v] = np.zeros((n_states, n_targets, n_tpoints), dtype=FC.DTYPE)
                results[v][:] = self.fixed_vars[v]
            elif v in self._solo.keys():
                results[v] = np.zeros((n_states, n_targets, n_tpoints), dtype=FC.DTYPE)
                results[v][:] = mdata[self.var(v)][:, None, None]
            else:
                results[v] = ires[vrs.index(v)]

        return results

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
        super().finalize(algo, verbosity)
        self._cmap = None
        self._solo = None
        self._weights = None
        self._N = None


class MultiHeightTimeseries(MultiHeightStates):
    """
    Multi-height timeseries states data.

    :group: input.states

    """

    RDICT = {"index_col": 0, "parse_dates": [0]}
