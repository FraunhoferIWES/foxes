from abc import abstractmethod

from .model import Model
from foxes.utils import all_subclasses


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
    def new(cls, profile_type, **kwargs):
        """
        Run-time profile factory.

        Parameters
        ----------
        profile_type: str
            The selected derived class name

        """

        if profile_type is None:
            return None

        allc = all_subclasses(cls)
        found = profile_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == profile_type:
                    return scls(**kwargs)

        else:
            estr = "Vertical profile type '{}' is not defined, available types are \n {}".format(
                profile_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
