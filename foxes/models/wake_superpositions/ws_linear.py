from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class WSLinear(WakeSuperposition):
    """
    Linear superposition of wind deficit results

    Attributes
    ----------
    scale_amb
        Flag for scaling wind deficit with ambient wind speed
        instead of waked wind speed
    lim_low
        Lower limit of the final waked wind speed
    lim_high
        Upper limit of the final waked wind speed


    """

    def __init__(
        self,
        scale_amb: bool = False,
        lim_low: float | None = None,
        lim_high: float | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        scale_amb
            Flag for scaling wind deficit with ambient wind speed
            instead of waked wind speed
        lim_low
            Lower limit of the final waked wind speed
        lim_high
            Upper limit of the final waked wind speed

        """
        super().__init__()

        self.scale_amb = scale_amb
        self.lim_low = lim_low
        self.lim_high = lim_high

    def __repr__(self) -> str:
        a = f"scale_amb={self.scale_amb}, lim_low={self.lim_low}, lim_high={self.lim_high}"
        return f"{type(self).__name__}({a})"

    def input_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are needed for running
        the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        input_vars
            The input variable names

        """
        return [FV.AMB_REWS] if self.scale_amb else [FV.REWS]

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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        st_sel
            The selection of targets, shape: (n_states, n_targets)
        variable
            The variable name for which the wake deltas applies
        wake_delta
            The original wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)
        wake_model_result
            The new wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, ...)

        Returns
        -------
        wdelta
            The updated wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(st_sel):
            scale = self.get_data(
                FV.AMB_REWS if self.scale_amb else FV.REWS,
                FC.STATE_TARGET_TPOINT,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            wake_delta[st_sel] += scale * wake_model_result

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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        variable
            The variable name for which the wake deltas applies
        wake_delta
            The wake deltas at targets, shape:
            (n_states, n_targets, n_tpoints)

        Returns
        -------
        final_wake_delta
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        w = wake_delta
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - tdata[FV.var2amb[variable]])
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - tdata[FV.var2amb[variable]])
        return w


class WSLinearLocal(WakeSuperposition):
    """
    Local linear superposition of wind deficit results

    Attributes
    ----------
    lim_low
        Lower limit of the final waked wind speed
    lim_high
        Upper limit of the final waked wind speed


    """

    def __init__(
        self, lim_low: float | None = None, lim_high: float | None = None
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        lim_low
            Lower limit of the final waked wind speed
        lim_high
            Upper limit of the final waked wind speed

        """
        super().__init__()
        self.lim_low = lim_low
        self.lim_high = lim_high

    def __repr__(self) -> str:
        a = f"lim_low={self.lim_low}, lim_high={self.lim_high}"
        return f"{type(self).__name__}({a})"

    def input_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are needed for running
        the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        input_vars
            The input variable names

        """
        return []

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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        st_sel
            The selection of targets, shape: (n_states, n_targets)
        variable
            The variable name for which the wake deltas applies
        wake_delta
            The original wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)
        wake_model_result
            The new wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, ...)

        Returns
        -------
        wdelta
            The updated wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )

        if np.any(st_sel):
            wake_delta[st_sel] += wake_model_result

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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        variable
            The variable name for which the wake deltas applies
        wake_delta
            The wake deltas at targets, shape:
            (n_states, n_targets, n_tpoints)

        Returns
        -------
        final_wake_delta
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        amb_results = tdata[FV.var2amb[variable]]
        w = wake_delta * amb_results
        if self.lim_low is not None:
            w = np.maximum(w, self.lim_low - amb_results)
        if self.lim_high is not None:
            w = np.minimum(w, self.lim_high - amb_results)
        return w
