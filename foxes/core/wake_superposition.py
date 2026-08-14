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
        pass

    @classmethod
    def new(cls, superp_type: str, *args: Any, **kwargs: Any) -> WakeSuperposition:
        """
        Run-time wake superposition model factory.

        Parameters
        ----------
        superp_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
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
        superp_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return new_instance(cls, superp_type, *args, **kwargs)
