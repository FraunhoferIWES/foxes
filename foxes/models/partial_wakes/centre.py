from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import foxes.variables as FV
from foxes.config import config

from .rotor_points import RotorPoints

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class PartialCentre(RotorPoints):
    """
    Partial wakes calculated only at the
    rotor centre point.


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
        return fdata[FV.TXYH][:, :, None], np.ones(1, dtype=config.dtype_double)

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

        if (
            downwind_index is None
            and len(rotor_res.shape) == 3
            and rotor_res.shape[:2]
            == (
                tdata.n_states,
                tdata.n_targets,
            )
        ):
            return np.einsum(
                "str,r->st",
                rotor_res,
                rotor_weights,
            )[:, :, None]

        elif (
            downwind_index is not None
            and len(rotor_res.shape) == 2
            and rotor_res.shape[0] == tdata.n_states
        ):
            return np.einsum(
                "sr,r->s",
                rotor_res,
                rotor_weights,
            )[:, None]

        else:
            return rotor_res
