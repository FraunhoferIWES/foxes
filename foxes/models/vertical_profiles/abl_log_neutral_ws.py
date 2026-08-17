from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import VerticalProfile
from foxes.utils.abl import neutral
import foxes.variables as FV
import foxes.constants as FC


class ABLLogNeutralWsProfile(VerticalProfile):
    """
    The neutral ABL wind speed log profile.

    Attributes
    ----------
    ustar_input
        Flag for using ustar as an input

    :group: models.vertical_profiles

    """

    def __init__(self, *args: Any, ustar_input: bool = False, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        args
            Additional arguments for VerticalProfile
        ustar_input
            Flag for using ustar as an input
        kwargs
            Additional arguments for VerticalProfile

        """
        super().__init__(*args, **kwargs)
        self.ustar_input = ustar_input

    def input_vars(self) -> list[str]:
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars
            The variable names

        """
        if self.ustar_input:
            return [FV.USTAR, FV.Z0]
        else:
            return [FV.WS, FV.H, FV.Z0]

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
        z0 = data[FV.Z0]
        if self.ustar_input:
            ustar = data[FV.USTAR]
        else:
            h0 = data[FV.H]
            ws = data[FV.WS]
            ustar = neutral.ustar(ws, h0, z0, kappa=FC.KAPPA)

        return np.asarray(neutral.calc_ws(heights, z0, ustar, kappa=FC.KAPPA))
