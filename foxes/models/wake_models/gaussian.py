import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel


class GaussianWakeModel(AxisymmetricWakeModel):
    """
    Abstract base class for Gaussian wake models.

    :group: models.wake_models

    """

    @abstractmethod
    def calc_amplitude_sigma(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_st_sel,)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        pass

    def calc_wakes_x_r(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        r,
    ):
        """
        Calculate wake deltas.

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_targets, n_yz_per_target)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_r_per_x)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        amsi, st_sel = self.calc_amplitude_sigma(
            algo, mdata, fdata, tdata, downwind_index, x
        )
        wdeltas = {}
        rsel = r[st_sel]
        for v in amsi.keys():
            ampld, sigma = amsi[v]
            wdeltas[v] = ampld[:, None] * np.exp(-0.5 * (rsel / sigma[:, None]) ** 2)

        return wdeltas, st_sel
