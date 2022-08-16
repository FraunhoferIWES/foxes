from abc import abstractmethod, ABCMeta


class Runner(metaclass=ABCMeta):
    """
    Abstract base class for runners.
    """

    @abstractmethod
    def run(self, func, args=tuple(), kwargs={}):
        """
        Runs the given function.

        Parameters
        ----------
        func : Function
            The function to be run
        args : tuple
            The function arguments
        kwargs : dict
            The function keyword arguments

        Returns
        -------
        results : Any
            The functions return value

        """
        pass
