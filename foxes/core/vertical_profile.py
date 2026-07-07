from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .model import Model
from foxes.utils import new_instance

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class VerticalProfile(Model):
    """
    Abstract base class for vertical profiles.

    :group: core

    """

    def load_chunk_data(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> None:
        """
        Load chunk-local data required for calculations.

        Vertical profiles operate on the provided chunk data directly and do
        not contribute additional chunk-local arrays.

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

        """
        return None

    @abstractmethod
    def input_vars(self) -> list[str]:
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return []

    @abstractmethod
    def calculate(self, tdata: TData, heights: np.ndarray) -> np.ndarray:
        """
        Run the profile calculation.

        Parameters
        ----------
        tdata: dict
            The target point data
        heights: numpy.ndarray
            The evaluation heights

        Returns
        -------
        results: numpy.ndarray
            The profile results, same
            shape as heights

        """
        pass

    @classmethod
    def new(
        cls,
        profile_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> "VerticalProfile":
        """
        Run-time vertical profile factory.

        Parameters
        ----------
        profile_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, profile_type, *args, **kwargs)
