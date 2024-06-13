import numpy as np

from foxes.core import AxialInductionModel


class BetzAxialInduction(AxialInductionModel):
    """
    The classic axial induction from 1D
    momentum theory

    Attributes
    ----------
    ct_max: float
        The maximal ct value

    :group: models.axial_induction

    """

    def __init__(self, ct_max=0.99999):
        """
        Constructor.

        Parameters
        ----------
        ct_max: float
            The maximal ct value

        """
        super().__init__()
        self.ct_max = ct_max

    def ct2a(self, ct):
        """
        Computes induction from ct

        Parameters
        ----------
        ct: numpy.ndarray or float
            The ct values

        Returns
        -------
        ct: numpy.ndarray or float
            The induction values

        """
        return 0.5 * (1 - np.sqrt(1 - np.minimum(ct, self.ct_max)))
