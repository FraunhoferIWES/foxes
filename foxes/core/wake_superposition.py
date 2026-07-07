from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from foxes.utils import new_instance

from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class WakeSuperposition(Model):
    """
    Abstract base class for wake superposition models.

    Note that it is a matter of the wake model
    if superposition models are used, or if the
    wake model computes the total wake result by
    other means.

    :group: core

    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @classmethod
    def new(cls, superp_type: str, *args: Any, **kwargs: Any) -> WakeSuperposition:
        """
        Run-time wake superposition model factory.

        Parameters
        ----------
        superp_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, superp_type, *args, **kwargs)


class WindVectorWakeSuperposition(Model):
    """
    Base class for wind vector superposition.

    Note that it is a matter of the wake model
    if superposition models are used, or if the
    wake model computes the total wake result by
    other means.

    :group: core

    """

    @abstractmethod
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
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)
        wake_model_result_uv: numpy.ndarray
            The new wind vector wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, 2, ...)

        Returns
        -------
        wdelta_uv: numpy.ndarray
            The updated wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        pass

    @abstractmethod
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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)

        Returns
        -------
        final_wake_delta_ws: numpy.ndarray
            The final wind speed wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)
        final_wake_delta_wd: numpy.ndarray
            The final wind direction wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        pass

    @classmethod
    def new(
        cls,
        superp_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> WindVectorWakeSuperposition:
        """
        Run-time wind wake superposition model factory.

        Parameters
        ----------
        superp_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, superp_type, *args, **kwargs)
