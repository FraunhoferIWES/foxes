import numpy as np

from foxes.core import States
import foxes.variables as FV
import foxes.constants as FC


class SingleStateStates(States):
    """
    A single uniform state.

    Parameters
    ----------
    ws : float
        The wind speed
    wd : float
        The wind direction
    ti : float, optional
        The TI value
    rho : float, optional
        The air density

    Attributes
    ----------
    ws : float
        The wind speed
    wd : float
        The wind direction
    ti : float
        The TI value
    rho : float
        The air density

    """

    def __init__(self, ws, wd, ti=None, rho=None):
        super().__init__()
        self.ws = ws
        self.wd = wd
        self.ti = ti
        self.rho = rho

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

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
        out = []
        if self.ws is not None:
            out.append(FV.WS)
        if self.wd is not None:
            out.append(FV.WD)
        if self.ti is not None:
            out.append(FV.TI)
        if self.rho is not None:
            out.append(FV.RHO)
        return out

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
        return np.ones((1, algo.n_turbines), dtype=FC.DTYPE)

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
        if self.ws is not None:
            pdata[FV.WS] = np.full(
                (pdata.n_states, pdata.n_points), self.ws, dtype=FC.DTYPE
            )
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
