import numpy as np

from foxes.core import VerticalProfile


class UniformProfile(VerticalProfile):
    """
    A profile with uniform values.

    Attributes
    ----------
    var: float
        The value

    :group: models.vertical_profiles

    """

    def __init__(self, variable):
        """
        Constructor

        Parameters
        ----------
        variable: float
            The value

        """
        super().__init__()
        self.var = variable

    def input_vars(self):
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return [self.var]

    def calculate(self, data, heights):
        """
        Run the profile calculation.

        Parameters
        ----------
        data: dict
            The input data
        heights: numpy.ndarray
            The evaluation heights

        Returns
        -------
        results: numpy.ndarray
            The profile results, same
            shape as heights

        """
        out = np.zeros_like(heights)
        out[:] = data[self.var]
        return out
