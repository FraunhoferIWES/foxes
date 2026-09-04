from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Any

from foxes.core import VerticalProfile


class DataProfile(VerticalProfile):
    """
    A profile based on numerical data.
    """

    def __init__(
        self,
        data_source: str | np.ndarray | pd.DataFrame,
        variable: str,
        col_z: str | int | None = None,
        col_var: str | int | None = None,
        pd_read_pars: dict[str, Any] | None = None,
        **interp_pars: Any,
    ) -> None:
        """
        Parameters
        ----------
        data_source
            The profile data
        variable
            The value
        col_z
            The column of z data
        col_var
            The column of variable data
        pd_read_pars
            Additional parameters for pandas.read_csv()
        interp_pars
            Additional parameters for interpolation
        """
        super().__init__()
        self.variable = variable
        self.interp_pars = interp_pars
        pd_read_pars = {} if pd_read_pars is None else pd_read_pars

        if isinstance(data_source, np.ndarray):
            col_z = col_z if col_z is not None else 0
            col_var = col_var if col_var is not None else -1
            self.data_z = data_source[col_z]
            self.data_v = data_source[col_var]
        else:
            if isinstance(data_source, pd.DataFrame):
                data = data_source
            else:
                data = pd.read_csv(data_source, **pd_read_pars)
            col_var = col_var if col_var is not None else variable
            self.data_v = data[col_var].to_numpy()
            if col_z is None:
                self.data_z = data.index.to_numpy()
            else:
                self.data_z = data[col_z].to_numpy()

        if not np.all(np.diff(self.data_z) > 0):
            inds = np.argsort(self.data_z)
            self.data_z = self.data_z[inds]
            self.data_v = self.data_v[inds]

    def input_vars(self) -> list[str]:
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars
            The variable names

        """
        return []

    def calculate(self, data: dict[str, Any], heights: np.ndarray) -> np.ndarray:
        """
        Run the profile calculation.

        Parameters
        ----------
        data
            The input data
        heights
            The evaluation heights

        Returns
        -------
        results
            The profile results, same
            shape as heights

        """
        return interp1d(self.data_z, self.data_v, **self.interp_pars)(heights)
