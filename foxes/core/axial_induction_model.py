from abc import abstractmethod

from .model import Model


class AxialInductionModel(Model):
    """
    Abstract base class for axial induction models

    :group: core

    """

    @abstractmethod
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
        pass
