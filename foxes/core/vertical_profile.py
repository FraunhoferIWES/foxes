from abc import abstractmethod

from .model import Model
from foxes.utils import new_instance


class VerticalProfile(Model):
    """
    Abstract base class for vertical profiles.

    :group: core

    """

    def load_chunk_data(self, algo, mdata, fdata, tdata):
        """
        Load chunk-local data required for calculations.

        Vertical profiles operate on the provided chunk data directly and do
        not contribute additional chunk-local arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        """
        return None

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
