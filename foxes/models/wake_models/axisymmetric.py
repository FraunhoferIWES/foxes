import numpy as np
from abc import abstractmethod

from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel


class AxisymmetricWakeModel(DistSlicedWakeModel):
    """
    Abstract base class for wake models
    that depend on (x, r) separately.

    The ability to evaluate multiple r values per x
    is used by the `PartialAxiwake` partial wakes model.

    :group: models.wake_models

    """

    @abstractmethod
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
        pass

    def calc_wakes_spsel_x_yz(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        yz,
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
        yz: numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_points, n_yz_per_x, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_yz_per_x)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        r = np.linalg.norm(yz, axis=-1)
        return self.calc_wakes_spsel_x_r(
            algo, mdata, fdata, pdata, states_source_turbine, x, r
        )
