import numpy as np

from foxes.core import WakeFrame
from foxes.utils import wd2uv
import foxes.variables as FV
import foxes.constants as FC


class RotorWD(WakeFrame):
    """
    Align the first axis for each rotor with the
    local normalized wind direction.

    Parameters
    ----------
    var_wd : str
        The wind direction variable

    Attributes
    ----------
    var_wd : str
        The wind direction variable

    """

    def __init__(self, var_wd=FV.WD):
        super().__init__()
        self.var_wd = var_wd

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

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

        Returns
        -------
        order : numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        n = np.mean(wd2uv(fdata[self.var_wd], axis=1), axis=-1)
        xy = fdata[FV.TXYH][:, :, :2]

        order = np.argsort(np.einsum("std,sd->st", xy, n), axis=-1)

        return order

    def get_wake_coos(self, algo, mdata, fdata, states_source_turbine, points):
        """
        Calculate wake coordinates.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        points : numpy.ndarray
            The evaluation points, shape: (n_states, n_points, 3)

        Returns
        -------
        wake_coos : numpy.ndarray
            The wake coordinates, shape: (n_states, n_points, 3)

        """
        n_states = mdata.n_states
        stsel = (np.arange(n_states), states_source_turbine)

        xyz = fdata[FV.TXYH][stsel]
        delta = points - xyz[:, None, :]
        del xyz

        wd = fdata[self.var_wd][stsel]

        nax = np.zeros((n_states, 3, 3), dtype=FC.DTYPE)
        n = nax[:, 0, :2]
        n[:] = wd2uv(wd, axis=-1)
        m = nax[:, 1, :2]
        m[:] = np.stack([-n[:, 1], n[:, 0]], axis=-1)
        nax[:, 2, 2] = 1
        del wd

        coos = np.einsum("spd,sad->spa", delta, nax)

        return coos
