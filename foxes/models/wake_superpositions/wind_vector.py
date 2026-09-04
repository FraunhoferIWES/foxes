from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import WindVectorWakeSuperposition
from foxes.utils import wd2uv, uv2wd, delta_wd
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class WindVectorLinear(WindVectorWakeSuperposition):
    """
    Linear superposition of wind deficit vector results
    """

    def __init__(self, scale_amb: bool = False) -> None:
        """
        Parameters
        ----------
        scale_amb
            Flag for scaling wind deficit with ambient wind speed
            instead of waked wind speed
        """
        super().__init__()
        self.scale_amb = scale_amb

    def __repr__(self) -> str:
        a = f"scale_amb={self.scale_amb}"
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

    def wdeltas_ws2uv(
        self,
        algo: Algorithm,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wdeltas: dict[str, np.ndarray],
        st_sel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Transform results from wind speed to wind vector data

        Parameters
        ----------
        algo
            The calculation algorithm
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value

        """
        if FV.AMB_UV not in tdata:
            tdata[FV.AMB_UV] = wd2uv(tdata[FV.AMB_WD], tdata[FV.AMB_WS])
        if FV.UV not in wdeltas:
            assert FV.WS in wdeltas, (
                f"{self.name}: Expecting '{FV.WS}' in wdeltas, got {list(wdeltas.keys())}"
            )
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
            ws0 = tdata[FV.AMB_WS][st_sel]
            wd0 = tdata[FV.AMB_WD][st_sel]
            dws = scale * wdeltas.pop(FV.WS)
            dwd = wdeltas.pop(FV.WD, 0)
            wdeltas[FV.UV] = wd2uv(wd0 + dwd, ws0 + dws) - tdata[FV.AMB_UV][st_sel]

        return wdeltas

    def wdeltas_uv2ws(
        self,
        algo: Algorithm,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wdeltas: dict[str, np.ndarray],
        st_sel: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Transform results from wind vector to wind speed data

        Parameters
        ----------
        algo
            The calculation algorithm
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wdeltas
            The wake deltas. Key: variable name str,
            value
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        Returns
        -------
        wdeltas
            The wake deltas. Key: variable name str,
            value

        """
        if FV.UV in wdeltas:
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
            ws0 = tdata[FV.AMB_WS][st_sel]
            wd0 = tdata[FV.AMB_WD][st_sel]
            uv = tdata[FV.AMB_UV][st_sel] + wdeltas.pop(FV.UV)
            wdeltas[FV.WD] = delta_wd(wd0, uv2wd(uv))
            wdeltas[FV.WS] = (np.linalg.norm(uv, axis=-1) - ws0) / scale

        return wdeltas

    def add_wake_vector(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        st_sel: np.ndarray,
        wake_delta_uv: np.ndarray,
        wake_model_result_uv: np.ndarray,
    ) -> np.ndarray:
        """
        Add a wake delta vector to previous wake deltas,
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
        wake_delta_uv
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)
        wake_model_result_uv
            The new wind vector wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, 2, ...)

        Returns
        -------
        wdelta_uv
            The updated wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """

        if np.any(st_sel):
            wake_delta_uv[st_sel] += wake_model_result_uv

        return wake_delta_uv

    def calc_final_wake_delta_uv(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wake_delta_uv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the final wind vector wake delta after adding all
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
        wake_delta_uv
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)

        Returns
        -------
        final_wake_delta_ws
            The final wind speed wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)
        final_wake_delta_wd
            The final wind direction wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        if FV.AMB_UV not in tdata:
            tdata[FV.AMB_UV] = wd2uv(tdata[FV.AMB_WD], tdata[FV.AMB_WS])

        uv = tdata[FV.AMB_UV] + wake_delta_uv
        dwd = delta_wd(tdata[FV.AMB_WD], uv2wd(uv))
        dws = np.linalg.norm(uv, axis=-1) - tdata[FV.AMB_WS]

        return dws, dwd
