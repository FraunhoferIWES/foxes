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
        pass

    def calc_wakes_x_yz(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        x,
        yz,
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
        yz: numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_targets, n_yz_per_target, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_st_sel, n_yz_per_target)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        r = np.linalg.norm(yz, axis=-1)
        return self.calc_wakes_x_r(algo, mdata, fdata, tdata, downwind_index, x, r)
