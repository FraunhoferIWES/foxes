from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import PartialWakesModel
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.wake_model import WakeModel


class RotorPoints(PartialWakesModel):
    """
    Partial wakes calculation directly by the
    rotor model.


    """

    def get_wake_points(
        self, algo: Algorithm, mdata: MData, fdata: FData
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the wake calculation points, and their
        weights.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data

        Returns
        -------
        rpoints
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights
            The target point weights, shape: (n_tpoints,)

        """
        return (
            algo.get_from_chunk_store(FC.ROTOR_POINTS, mdata=mdata),
            algo.get_from_chunk_store(FC.ROTOR_WEIGHTS, mdata=mdata),
        )

    def map_rotor_results(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        variable: str,
        rotor_res: np.ndarray,
        rotor_weights: np.ndarray,
        downwind_index: int | None = None,
    ) -> np.ndarray:
        """
        Map ambient rotor point results onto target points.

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
            The variable name to map
        rotor_res
            The results at rotor points, shape:
            (n_states, n_turbines, n_rotor_points) if downwind_index is None,
            otherwise shape: (n_states, n_rotor_points)
        rotor_weights
            The rotor point weights, shape: (n_rotor_points,)
        downwind_index
            The downwind index of the updated turbine,
            if None, maps for all turbines

        Returns
        -------
        res
            The mapped results at target points, shape:
            (n_states, n_targets, n_tpoints) if downwind_index is None,
            otherwise shape: (n_states, n_tpoints)

        """
        return rotor_res

    def finalize_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        rpoint_weights: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
        downwind_index: int,
    ) -> dict[str, np.ndarray]:
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

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
        rpoint_weights
            The rotor point weights, shape: (n_rotor_points,)
        wake_deltas
            The wake deltas. Key: variable name,
            value: np.ndarray of shape
            (n_states, n_turbines, n_tpoints)
        wmodel
            The wake model
        downwind_index
            The index in the downwind order

        Returns
        -------
        final_wake_deltas
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        wdel: dict[str, np.ndarray] = {
            v: d[:, downwind_index, None].copy() if d.shape[1] > 1 else d[:, 0, None]
            for v, d in wake_deltas.items()
        }
        wmodel.finalize_wake_deltas(algo, mdata, fdata, tdata, wdel)

        return {v: d[:, 0] for v, d in wdel.items()}
