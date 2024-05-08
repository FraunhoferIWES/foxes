import numpy as np

from foxes.core import WakeFrame
from foxes.utils import wd2uv
import foxes.variables as FV
import foxes.constants as FC


class RotorWD(WakeFrame):
    """
    Align the first axis for each rotor with the
    local normalized wind direction.

    Attributes
    ----------
    var_wd: str
        The wind direction variable

    :group: models.wake_frames

    """

    def __init__(self, var_wd=FV.WD):
        """
        Constructor.

        Parameters
        ----------
        var_wd: str
            The wind direction variable

        """
        super().__init__()
        self.var_wd = var_wd

    def calc_order(self, algo, mdata, fdata):
        """ "
        Calculates the order of turbine evaluation.

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

        Returns
        -------
        order: numpy.ndarray
            The turbine order, shape: (n_states, n_turbines)

        """
        n = np.mean(wd2uv(fdata[self.var_wd], axis=1), axis=-1)
        xy = fdata[FV.TXYH][:, :, :2]
        order = np.argsort(np.einsum("std,sd->st", xy, n), axis=-1)

        return order

    def get_wake_coos(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
    ):
        """
        Calculate wake coordinates of rotor points.

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
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order

        Returns
        -------
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        n_states = tdata.n_states
        targets = tdata[FC.TARGETS]

        xyz = fdata[FV.TXYH][:, downwind_index]
        delta = targets - xyz[:, None, None, :]
        del xyz

        wd = fdata[self.var_wd][:, downwind_index]

        nax = np.zeros((n_states, 3, 3), dtype=FC.DTYPE)
        n = nax[:, 0, :2]
        n[:] = wd2uv(wd, axis=-1)
        m = nax[:, 1, :2]
        m[:] = np.stack([-n[:, 1], n[:, 0]], axis=-1)
        nax[:, 2, 2] = 1
        del wd

        coos = np.einsum("stpd,sad->stpa", delta, nax)

        return coos

    def get_centreline_points(self, algo, mdata, fdata, downwind_index, x):
        """
        Gets the points along the centreline for given
        values of x.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        downwind_index: int
            The index in the downwind order
        x: numpy.ndarray
            The wake frame x coordinates, shape: (n_states, n_points)

        Returns
        -------
        points: numpy.ndarray
            The centreline points, shape: (n_states, n_points, 3)

        """
        wd = fdata[self.var_wd][:, downwind_index]
        n = np.append(wd2uv(wd, axis=-1), np.zeros_like(wd)[:, None], axis=-1)

        xyz = fdata[FV.TXYH][:, downwind_index]
        return xyz[:, None, :] + x[:, :, None] * n[:, None, :]
