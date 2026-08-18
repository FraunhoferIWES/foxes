from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
import pandas as pd
from scipy.interpolate import interpn
from typing import TYPE_CHECKING, Any

from foxes.core import TurbineModel
from foxes.utils import PandasFileHelper
from foxes.config import config, get_input_path

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class TableFactors(TurbineModel):
    """
    Multiplies variables by factors from a
    two dimensional table.

    The column names are expected to be numbers
    that represent the col_var variable.

    Attributes
    ----------
    data_source
        Either path to a file or data
    row_var
        The row-wise variable
    col_var
        The column-wise variable
    ovars
        The variables onto which the factors
        are multiplied


    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        row_var: str,
        col_var: str,
        output_vars: list[str],
        pd_file_read_pars: dict[str, Any] = {},
        **ipars: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            Either path to a file or data
        row_var
            The row-wise variable
        col_var
            The column-wise variable
        output_vars
            The variables onto which the factors
            are multiplied
        pd_file_read_pars
            Parameters for pandas file reading
        ipars
            Parameters for scipy.interpolate.interpn

        """
        super().__init__()

        self.data_source = data_source
        self.row_var = row_var
        self.col_var = col_var
        self.ovars = output_vars
        self._rpars = pd_file_read_pars
        self._ipars = ipars

        self._rvals: np.ndarray | None = None
        self._cvals: np.ndarray | None = None
        self._data: np.ndarray | None = None

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
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

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        data_df: pd.DataFrame
        if isinstance(self.data_source, pd.DataFrame):
            data_df = self.data_source
        else:
            fpath = get_input_path(self.data_source)
            if verbosity > 0:
                print(f"{self.name}: Reading file {fpath}")
            rpars = dict(index_col=0)
            rpars.update(self._rpars)
            data_df = PandasFileHelper.read_file(fpath, **rpars)

        self._rvals = data_df.index.to_numpy(config.dtype_double)
        self._cvals = data_df.columns.to_numpy(config.dtype_double)
        self._data = data_df.to_numpy(config.dtype_double)
        return loaded_data

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
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
        st_sel: slice or array of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        self.ensure_output_vars(algo, fdata)
        rvals = self._rvals
        cvals = self._cvals
        data = self._data
        assert rvals is not None and cvals is not None and data is not None

        n_sel = np.size(fdata[self.row_var][st_sel])
        qts: np.ndarray = np.zeros((n_sel, 2), dtype=config.dtype_double)
        qts[:, 0] = np.asarray(fdata[self.row_var][st_sel]).reshape(n_sel)
        qts[:, 1] = np.asarray(fdata[self.col_var][st_sel]).reshape(n_sel)

        try:
            factors = interpn((rvals, cvals), data, qts, **self._ipars)
        except ValueError as e:
            print(f"\nDATA       : ({self.row_var}, {self.col_var})")
            print(
                f"DATA BOUNDS: ({np.min(rvals)}, {np.min(cvals)}) -- ({np.max(rvals)}, {np.max(cvals)})"
            )
            print(
                f"VALUE BOUNDS: ({np.min(qts[:, 0]):.4f}, {np.min(qts[:, 1]):.4f}) -- ({np.max(qts[:, 0]):.4f}, {np.max(qts[:, 1]):.4f})\n"
            )
            raise e

        for v in self.output_farm_vars(algo):
            fdata[v][st_sel] *= factors

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
