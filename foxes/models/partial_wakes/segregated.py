from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, cast

from foxes.core import PartialWakesModel, TData
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import Model
    from foxes.core.rotor_model import RotorModel
    from foxes.core.wake_model import WakeModel


class PartialSegregated(PartialWakesModel):
    """
    Add the averaged wake effects to the separately
    averaged ambient rotor results

    Attributes
    ----------
    rotor_model
        The rotor model, default is the one from the algorithm
    grotor
        The grid rotor model

    :group: models.partial_wakes

    """

    def __init__(self, rotor_model: RotorModel) -> None:
        """
        Constructor.

        Parameters
        ----------
        rotor_model
            The rotor model for wake averaging

        """
        super().__init__()

        self.rotor = rotor_model
        self.YZ = self.var("YZ")
        self.W = self.var(FV.WEIGHT)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rotor_model={self.rotor.name})"

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return super().sub_models() + [self.rotor]

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
            self.rotor.get_rotor_points(algo, mdata, fdata),
            self.rotor.rotor_point_weights(),
        )

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
        n_states = fdata.n_states
        assert n_states is not None
        n_rotor_points = len(rpoint_weights)
        gweights = tdata[FC.TWEIGHTS]

        wdel = {v: d[:, downwind_index, None].copy() for v, d in wake_deltas.items()}
        htdata = cast(TData, tdata.get_slice([FC.TURBINE], np.s_[downwind_index]))
        wmodel.finalize_wake_deltas(algo, mdata, fdata, htdata, wdel)

        for v in wdel.keys():
            hdel: np.ndarray = np.zeros(
                (n_states, n_rotor_points), dtype=config.dtype_double
            )
            hdel[:] = np.einsum("sp,p->s", wdel[v][:, 0], gweights)[:, None]
            wdel[v] = hdel

        return wdel
