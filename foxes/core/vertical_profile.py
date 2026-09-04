from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .model import Model
from foxes.utils import new_instance

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class VerticalProfile(Model):
    """
    Abstract base class for vertical profiles.
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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
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
        vars
            The variable names

        """
        return []

    @abstractmethod
    def calculate(self, tdata: TData, heights: np.ndarray) -> np.ndarray:
        """
        Run the profile calculation.

        Parameters
        ----------
        tdata
            The target point data
        heights
            The evaluation heights

        Returns
        -------
        results
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
        profile_type
            The selected derived class name
        args
            Additional parameters for the constructor
        kwargs
            Additional parameters for the constructor

        """
        return cast(VerticalProfile, new_instance(cls, profile_type, *args, **kwargs))
