from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import VerticalProfile
from foxes.utils.abl import stable
import foxes.variables as FV
import foxes.constants as FC


class ABLLogStableWsProfile(VerticalProfile):
    """
    The stable ABL wind speed log profile.

    Attributes
    ----------
    ustar_input
        Flag for using ustar as an input


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
            return [FV.USTAR, FV.Z0, FV.MOL]
        else:
            return [FV.WS, FV.H, FV.Z0, FV.MOL]

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
        mol = data[FV.MOL]

        if self.ustar_input:
            ustar = data[FV.USTAR]
        else:
            ws = data[FV.WS]
            h0 = data[FV.H]
            ustar = stable.ustar(ws, h0, z0, mol, kappa=FC.KAPPA)
        psi = stable.psi(heights, mol)

        return np.asarray(stable.calc_ws(heights, z0, ustar, psi, kappa=FC.KAPPA))
