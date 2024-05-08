import numpy as np
import pandas as pd
from scipy.interpolate import interpn

from foxes.core import TurbineModel
from foxes.utils import PandasFileHelper
import foxes.constants as FC


class TableFactors(TurbineModel):
    """
    Multiplies variables by factors from a
    two dimensional table.

    The column names are expected to be numbers
    that represent the col_var variable.

    Attributes
    ----------
    data_source: str or pandas.DataFrame
        Either path to a file or data
    row_var: str
        The row-wise variable
    col_var: str
        The column-wise variable
    ovars: list of str
        The variables onto which the factors
        are multiplied

    :group: models.turbine_models

    """

    def __init__(
        self,
        data_source,
        row_var,
        col_var,
        output_vars,
        pd_file_read_pars={},
        **ipars,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            Either path to a file or data
        row_var: str
            The row-wise variable
        col_var: str
            The column-wise variable
        output_vars: list of str
            The variables onto which the factors
            are multiplied
        pd_file_read_pars: dict
            Parameters for pandas file reading
        ipars: dict, optional
            Parameters for scipy.interpolate.interpn

        """
        super().__init__()

        self.data_source = data_source
        self.row_var = row_var
        self.col_var = col_var
        self.ovars = output_vars
        self._rpars = pd_file_read_pars
        self._ipars = ipars

        self._cvals = None
        self._data = None

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
        return self.ovars

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
        super().initialize(algo, verbosity)

        if isinstance(self.data_source, pd.DataFrame):
            self._data = self.data_source
        else:
            if verbosity > 0:
                print(f"{self.name}: Reading file {self.data_source}")
            rpars = dict(index_col=0)
            rpars.update(self._rpars)
            self._data = PandasFileHelper.read_file(self.data_source, **rpars)

        self._rvals = self._data.index.to_numpy(FC.DTYPE)
        self._cvals = self._data.columns.to_numpy(FC.DTYPE)
        self._data = self._data.to_numpy(FC.DTYPE)

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
        n_sel = np.sum(st_sel)
        qts = np.zeros((n_sel, 2), dtype=FC.DTYPE)
        qts[:, 0] = fdata[self.row_var][st_sel]
        qts[:, 1] = fdata[self.col_var][st_sel]

        try:
            factors = interpn(
                (self._rvals, self._cvals), self._data, qts, **self._ipars
            )
        except ValueError as e:
            print(f"\nDATA       : ({self.row_var}, {self.col_var})")
            print(
                f"DATA BOUNDS: ({np.min(self._rvals)}, {np.min(self._cvals)}) -- ({np.max(self._rvals)}, {np.max(self._cvals)})"
            )
            print(
                f"VALUE BOUNDS: ({np.min(qts[:, 0]):.4f}, {np.min(qts[:, 1]):.4f}) -- ({np.max(qts[:, 0]):.4f}, {np.max(qts[:, 1]):.4f})\n"
            )
            raise e

        for v in self.output_farm_vars(algo):
            fdata[v][st_sel] *= factors

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
