from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import WakeSuperposition
from foxes.core.wake_superposition import get_ws_scale
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class WSQuadratic(WakeSuperposition):
    """Quadratic superposition of wind deficit results."""

    def __init__(
        self,
        scale_amb: bool = False,
        scale_target: bool = False,
        lim_low: float | None = None,
        lim_high: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        scale_amb
            Flag for selecting ambient instead of waked wind speed.
        scale_target
            Flag for selecting target instead of source turbine wind speed.
        lim_low
            Lower limit of the final waked wind speed.
        lim_high
            Upper limit of the final waked wind speed.
        """
        super().__init__()
        self.scale_amb = scale_amb
        self.scale_target = scale_target
        self.lim_low = lim_low
        self.lim_high = lim_high

    def __repr__(self) -> str:
        args = (
            f"scale_amb={self.scale_amb}, scale_target={self.scale_target}, "
            f"lim_low={self.lim_low}, lim_high={self.lim_high}"
        )
        return f"{type(self).__name__}({args})"

    def input_farm_vars(self, algo: Algorithm) -> list[str]:
        """Return farm variables required for wake scaling."""
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
        """Add a selected wake contribution to the accumulated deficit."""
        if variable not in [FV.REWS, FV.REWS2, FV.REWS3, FV.WS]:
            raise ValueError(
                f"Superposition '{self.name}': Expecting wind speed variable, got {variable}"
            )
        if np.any(st_sel):
            scale = get_ws_scale(
                self,
                self.scale_amb,
                self.scale_target,
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
            )
            wake_delta[st_sel] += (scale * wake_model_result) ** 2
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
        """Apply optional wind-speed limits to the accumulated deficit."""
        result = -np.sqrt(wake_delta)
        ambient = tdata[FV.var2amb[variable]]
        if self.lim_low is not None:
            result = np.maximum(result, self.lim_low - ambient)
        if self.lim_high is not None:
            result = np.minimum(result, self.lim_high - ambient)
        return result
