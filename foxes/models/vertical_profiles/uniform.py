from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import VerticalProfile


class UniformProfile(VerticalProfile):
    """
    A profile with uniform values.

    Attributes
    ----------
    var
        The value


    """

    def __init__(self, variable: str) -> None:
        """
        Constructor

        Parameters
        ----------
        variable
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
        vars
            The variable names

        """
        return [self.variable]

    def calculate(self, data: dict[str, Any], heights: np.ndarray) -> np.ndarray:
        """
        Run the profile calculation.

        Parameters
        ----------
        data
            The input data
        heights
            The evaluation heights

        Returns
        -------
        results
            The profile results, same
            shape as heights

        """
        out = np.zeros_like(heights)
        out[:] = data[self.variable]
        return out
