from foxes.core import Runner

class SerialRunner(Runner):
    """
    Class for serial function execution.
    """

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
        return func(*args, **kwargs)
