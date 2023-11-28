import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from foxes.core import VerticalProfile


class DataProfile(VerticalProfile):
    """
    A profile based on numerical data.

    Attributes
    ----------
    var: float
        The value
    data_z: numpy.ndarray
        The z values, shape: (n_z,)
    data_v: numpy.ndarray
        The variable values, shape: (n_z,)
    interp_pars: dict
        Additional parameters for interpolation

    :group: models.vertical_profiles

    """

    def __init__(
        self,
        data_source,
        variable,
        col_z=None,
        col_var=None,
        pd_read_pars={},
        **interp_pars
    ):
        """
        Constructor

        Parameters
        ----------
        data_source: str or numpy.ndarray or pandas.DataFrame
            The profile data
        variable: float
            The value
        col_z: str or int, optional
            The column of z data
        col_var: str or int, optional
            The column of variable data
        pd_read_pars: dict
            Additional parameters for pandas.read_csv()
        interp_pars: dict, optional
            Additional parameters for interpolation

        """
        super().__init__()
        self.var = variable
        self.interp_pars = interp_pars

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

    def input_vars(self):
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return []

    def calculate(self, data, heights):
        """
        Run the profile calculation.

        Parameters
        ----------
        data: dict
            The input data
        heights: numpy.ndarray
            The evaluation heights

        Returns
        -------
        results: numpy.ndarray
            The profile results, same
            shape as heights

        """
        return interp1d(self.data_z, self.data_v, **self.interp_pars)(heights)
