from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import VerticalProfile


class UniformProfile(VerticalProfile):
    """
    A profile with uniform values.

    Attributes
    ----------
    var: float
        The value

    :group: models.vertical_profiles

    """

    def __init__(self, variable: str) -> None:
        """
        Constructor

        Parameters
        ----------
        variable: float
            The value

        """
        super().__init__()
        self.variable = variable

    def input_vars(self) -> list[str]:
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return [self.variable]

    def calculate(self, data: dict[str, Any], heights: np.ndarray) -> np.ndarray:
        """
        Run the profile calculation.

        Parameters
        ----------
        data: dict
            The input data
        heights: numpy.ndarray
            The evaluation heights

        Returns
        -------
        results: numpy.ndarray
            The profile results, same
            shape as heights

        """
        out = np.zeros_like(heights)
        out[:] = data[self.variable]
        return out
