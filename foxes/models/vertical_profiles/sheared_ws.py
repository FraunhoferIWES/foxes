import numpy as np

from foxes.core import VerticalProfile
from foxes.utils import abl
import foxes.variables as FV


class ShearedProfile(VerticalProfile):
    """
    A wind shear profile, based on a shear exponent.

    :group: models.vertical_profiles

    """

    def input_vars(self):
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        return [FV.WS, FV.H, FV.SHEAR]

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

        ws = np.zeros_like(heights)
        ws[:] = data[FV.WS]

        h0 = np.zeros_like(heights)
        h0[:] = data[FV.H]

        shear = np.zeros_like(heights)
        shear[:] = data[FV.SHEAR]

        out = np.zeros_like(heights)
        out[:] = abl.sheared.calc_ws(heights, h0, ws, shear)

        return out
