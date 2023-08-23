import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.axisymmetric import AxisymmetricWakeModel


class GaussianWakeModel(AxisymmetricWakeModel):
    """
    Abstract base class for Gaussian wake models.

    :group: models.wake_models

    """

    @abstractmethod
    def calc_amplitude_sigma_spsel(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
    ):
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_sp_sel,)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        pass

    def calc_wakes_spsel_x_r(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        r,
    ):
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)
        r: numpy.ndarray
            The radial values for each x value, shape:
            (n_states, n_points, n_r_per_x, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_r_per_x)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        amsi, sp_sel = self.calc_amplitude_sigma_spsel(
            algo, mdata, fdata, pdata, states_source_turbine, x
        )
        wdeltas = {}
        rsel = r[sp_sel]
        for v in amsi.keys():
            ampld, sigma = amsi[v]
            wdeltas[v] = ampld[:, None] * np.exp(-0.5 * (rsel / sigma[:, None]) ** 2)

        return wdeltas, sp_sel
