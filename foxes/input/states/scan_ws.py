import numpy as np

from foxes.core import States
import foxes.variables as FV
import foxes.constants as FC


class ScanWS(States):
    """
    A given list of wind speeds, all other variables are fixed.

    Parameters
    ----------
    ws_list : array_like
        The wind speed values
    wd : float
        The wind direction
    ti : float, optional
        The TI value
    rho : float, optional
        The air density

    Parameters
    ----------
    wd : float
        The wind direction
    ti : float
        The TI value
    rho : float
        The air density

    """

    def __init__(self, ws_list, wd, ti=None, rho=None):
        super().__init__()

        self._wsl = np.array(ws_list)
        self.N = len(ws_list)
        self.wd = wd
        self.ti = ti
        self.rho = rho

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
        self.WS = f"{self.name}_ws"

        idata = super().model_input_data(algo)
        idata["data_vars"][self.WS] = ((FV.STATE,), self._wsl)

        del self._wsl

        return idata

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.N

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
        pvars = [FV.WS]
        if self.wd is not None:
            pvars.append(FV.WD)
        if self.ti is not None:
            pvars.append(FV.TI)
        if self.rho is not None:
            pvars.append(FV.RHO)
        return pvars

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
        return np.full((self.N, algo.n_turbines), 1.0 / self.N, dtype=FC.DTYPE)

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
        pdata[FV.WS][:] = mdata[self.WS][:, None]

        if self.wd is not None:
            pdata[FV.WD] = np.full(
                (pdata.n_states, pdata.n_points), self.wd, dtype=FC.DTYPE
            )
        if self.ti is not None:
            pdata[FV.TI] = np.full(
                (pdata.n_states, pdata.n_points), self.ti, dtype=FC.DTYPE
            )
        if self.rho is not None:
            pdata[FV.RHO] = np.full(
                (pdata.n_states, pdata.n_points), self.rho, dtype=FC.DTYPE
            )

        return {v: pdata[v] for v in self.output_point_vars(algo)}
