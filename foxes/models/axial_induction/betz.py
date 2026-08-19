from __future__ import annotations

import numpy as np

from foxes.core import AxialInductionModel


class BetzAxialInduction(AxialInductionModel):
    """
    The classic axial induction from 1D
    momentum theory

    Attributes
    ----------
    ct_max
        The maximal ct value


    """

    def __init__(self, ct_max: float = 0.99999) -> None:
        """
        Constructor.

        Parameters
        ----------
        ct_max
            The maximal ct value

        """
        super().__init__()
        self.ct_max = ct_max

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
        return 0.5 * (1 - np.sqrt(1 - np.minimum(ct, self.ct_max)))
