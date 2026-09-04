from __future__ import annotations

import numpy as np

from foxes.core import AxialInductionModel


class MadsenAxialInduction(AxialInductionModel):
    """
    Computes the induction factor through polynomial
    fit, extending validity for high ct values

    Notes
    -----
    Reference:
    Helge Aagaard Madsen, Torben Juul Larsen, Georg Raimund Pirrung, Ang Li, and Frederik Zahle
    "Implementation of the blade element momentum model on a polar grid and its aeroelastic load impact"
    https://doi.org/10.5194/wes-5-1-2020
    """

    def __init__(
        self, k1: float = 0.2460, k2: float = 0.0586, k3: float = 0.0883
    ) -> None:
        """
        Parameters
        ----------
        k1
            Model coefficient
        k2
            Model coefficient
        k3
            Model coefficient
        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def ct2a(self, ct: np.ndarray | float) -> np.ndarray | float:
        """
        Computes induction from ct

        Parameters
        ----------
        ct
            The ct values

        Returns
        -------
        ct
            The induction values

        """
        return self.k1 * ct + self.k2 * ct**2 + self.k3 * ct**3
