from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import WakeSuperposition
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class TIMax(WakeSuperposition):
    """
    Max wake superposition for TI.

    Attributes
    ----------
    superp_to_amb: str
        The method for combining ambient with wake deltas:
        linear or quadratic

    :group: models.wake_superpositions

    """

    def __init__(self, superp_to_amb: str = "quadratic") -> None:
        """
        Constructor.

        Parameters
        ----------
        superp_to_amb: str
            The method for combining ambient with wake deltas:
            linear or quadratic

        """
        super().__init__()
        self.superp_to_amb = superp_to_amb

    def __repr__(self) -> str:
        return f"{type(self).__name__}(superp_to_amb={self.superp_to_amb})"

    def add_wake(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        st_sel: np.ndarray,
        variable: str,
        wake_delta: np.ndarray,
        wake_model_result: np.ndarray,
    ) -> np.ndarray:
        """
        Add a wake delta to previous wake deltas,
        at rotor points.

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
            in the downwind order
        st_sel: numpy.ndarray of bool
            The selection of targets, shape: (n_states, n_targets)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, ...)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        if variable != FV.TI:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wake variable {FV.TI}, got {variable}"
            )

        wake_delta[st_sel] = np.maximum(wake_model_result, wake_delta[st_sel])
        return wake_delta

    def calc_final_wake_delta(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        variable: str,
        wake_delta: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the final wake delta after adding all
        contributions.

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
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The wake deltas at targets, shape:
            (n_states, n_targets, n_tpoints)

        Returns
        -------
        final_wake_delta: numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        # linear superposition to ambient:
        if self.superp_to_amb == "linear":
            return wake_delta

        # quadratic superposition to ambient:
        elif self.superp_to_amb == "quadratic":
            amb_results = tdata[FV.var2amb[variable]]
            return np.sqrt(wake_delta**2 + amb_results**2) - amb_results

        # unknown ti delta:
        else:
            raise ValueError(
                f"Unknown superp_to_amb = '{self.superp_to_amb}', valid choices: linear, quadratic"
            )
