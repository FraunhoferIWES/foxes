from abc import abstractmethod

from foxes.core.model import Model
from foxes.tools import all_subclasses

class VerticalProfile(Model):

    @abstractmethod
    def input_vars(self):
        return []

    @abstractmethod
    def calculate(self, data, heights):
        pass

    @classmethod
    def new(cls, profile_type, **kwargs):
        """
        Run-time profile factory.

        Parameters
        ----------
        profile_type : str
            The selected derived class name

        """

        if profile_type is None:
            return None

        allc  = all_subclasses(cls)
        found = profile_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == profile_type:
                    return scls(**kwargs)

        else:
            estr = "Vertical profile type '{}' is not defined, available types are \n {}".format(
                profile_type, sorted([ i.__name__ for i in allc]))
            raise KeyError(estr)
