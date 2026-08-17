from __future__ import annotations

import numpy as np
from abc import abstractmethod
from typing import TYPE_CHECKING

from foxes.models.wake_models.dist_sliced import DistSlicedWakeModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


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
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        r: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        r
            The radial values for each x value, shape:
            (n_states, n_targets, n_yz_per_target)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        pass

    def calc_wakes_x_yz(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        yz: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        yz
            The yz values for each x value, shape:
            (n_states, n_targets, n_yz_per_target, 2)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        r = np.linalg.norm(yz, axis=-1)
        return self.calc_wakes_x_r(algo, mdata, fdata, tdata, downwind_index, x, r)
