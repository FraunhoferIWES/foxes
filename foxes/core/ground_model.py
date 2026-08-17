from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from foxes.utils import new_instance

from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.partial_wakes_model import PartialWakesModel
    from foxes.core.wake_model import WakeModel


class GroundModel(Model):
    """
    Base class for ground models.
    """

    def new_farm_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wmodel: WakeModel,
        pwake: PartialWakesModel,
    ) -> dict[str, np.ndarray]:
        """
        Create new initial wake deltas filled with zeros.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target point data.
        wmodel
            The wake model.
        pwake
            The partial wakes model.

        Returns
        -------
        wake_deltas
            A dictionary keyed by variable name. Values are zero-filled wake
            deltas with shape ``(n_states, n_turbines, n_tpoints, ...)``.

        """
        return pwake.new_wake_deltas(algo, mdata, fdata, tdata, wmodel)

    def contribute_to_farm_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
        pwake: PartialWakesModel,
    ) -> None:
        """
        Modify wake deltas at target points using contributions from wake source
        turbines.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target-point data.
        downwind_index
            The index of the wake-causing turbine in the downwind order.
        wake_deltas
            The wake deltas. Keys are variable names and values are arrays with
            shape ``(n_states, n_targets, n_tpoints, ...)``.
        wmodel
            The wake model.
        pwake
            The partial wakes model.

        """
        pwake.contribute(algo, mdata, fdata, tdata, downwind_index, wake_deltas, wmodel)

    def finalize_farm_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        rpoint_weights: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
        downwind_index: int,
        pwake: PartialWakesModel,
    ) -> dict[str, np.ndarray]:
        """
        Update wake deltas at the selected target downwind index.

        This modifies ``wake_deltas`` in place.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target-point data.
        rpoint_weights
            The rotor-point weights with shape ``(n_rotor_points,)``.
        wake_deltas
            The wake deltas. Keys are variable names and values are arrays with
            shape ``(n_states, n_turbines, n_tpoints)``.
        wmodel
            The wake model.
        downwind_index
            The index in the downwind order.

        Returns
        -------
        final_wake_deltas
            The final wake deltas at the selected downwind turbines. Keys are
            variable names and values are arrays with shape
            ``(n_states, n_rotor_points)``.

        """
        return pwake.finalize_wakes(
            algo,
            mdata,
            fdata,
            tdata,
            rpoint_weights,
            wake_deltas,
            wmodel,
            downwind_index,
        )

    def new_point_wake_deltas(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wmodel: WakeModel,
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

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
        wmodel
            The wake model

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, tdata)

    def contribute_to_point_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        wake_deltas
            The wake deltas. Key: variable name,
            values are arrays with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel
            The wake model

        """
        wake_frame = algo.wake_frame
        wcoos = wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

    def finalize_point_wakes(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Finalize the wake calculation.

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
        wake_deltas
            The wake deltas object at the selected target
            turbines. Keys are variable names and values are arrays
            with shape (n_states, n_targets, n_tpoints)

        """
        wmodel.finalize_wake_deltas(algo, mdata, fdata, tdata, wake_deltas)

    @classmethod
    def new(cls, ground_type: str, *args: Any, **kwargs: Any) -> GroundModel:
        """
        Run-time ground model factory.

        Parameters
        ----------
        ground_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        obj = new_instance(cls, ground_type, *args, **kwargs)
        if obj is None:
            raise ValueError(f"Ground model '{ground_type}' not found")
        return cast(GroundModel, obj)
