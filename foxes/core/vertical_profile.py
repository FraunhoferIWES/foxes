from abc import abstractmethod

from .model import Model
from foxes.utils import new_instance


class VerticalProfile(Model):
    """
    Abstract base class for vertical profiles.

    :group: core

    """

    @abstractmethod
    def input_vars(self):
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return []

    @abstractmethod
    def calculate(self, tdata, heights):
        """
        Run the profile calculation.

        Parameters
        ----------
        tdata: dict
            The target point data
        heights: numpy.ndarray
            The evaluation heights

        Returns
        -------
        results: numpy.ndarray
            The profile results, same
            shape as heights

        """
        pass

    @classmethod
    def new(cls, profile_type, *args, **kwargs):
        """
        Run-time vertical profile factory.

        Parameters
        ----------
        profile_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for the constructor
        kwargs: dict, optional
            Additional parameters for the constructor

        """
        return new_instance(cls, profile_type, *args, **kwargs)
