from abc import abstractmethod
from typing import Any

import numpy as np

from foxes.utils import new_instance
from .model import Model


class AxialInductionModel(Model):
    """
    Abstract base class for axial induction models

    :group: core

    """

    @abstractmethod
    def ct2a(self, ct: np.ndarray | float) -> np.ndarray | float:
        """
        Compute induction from the thrust coefficient.

        Parameters
        ----------
        ct
            The thrust coefficient values.

        Returns
        -------
        ct
            The induction values.

        """
        pass

    @classmethod
    def new(
        cls,
        induction_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> "AxialInductionModel":
        """
        Create an axial induction model instance at runtime.

        Parameters
        ----------
        induction_type
            The selected derived class name.
        args
            Additional positional arguments for the constructor.
        kwargs
            Additional keyword arguments for the constructor.

        """
        return new_instance(cls, induction_type, *args, **kwargs)
