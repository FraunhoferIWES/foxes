from abc import abstractmethod

from foxes.utils import new_instance
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

    @classmethod
    def new(cls, induction_type, *args, **kwargs):
        """
        Run-time axial induction model factory.

        Parameters
        ----------
        induction_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, induction_type, *args, **kwargs)
