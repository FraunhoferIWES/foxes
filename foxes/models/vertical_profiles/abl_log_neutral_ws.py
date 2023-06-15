from foxes.core import VerticalProfile
from foxes.utils.abl import neutral
import foxes.constants as FC
import foxes.variables as FV


class ABLLogNeutralWsProfile(VerticalProfile):
    """
    The neutral ABL wind speed log profile.

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
        return [FV.WS, FV.H, FV.Z0]

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
        z0 = data[FV.Z0]
        h0 = data[FV.H]
        ws = data[FV.WS]

        ustar = neutral.ustar(ws, h0, z0, kappa=FC.KAPPA)

        return neutral.calc_ws(heights, z0, ustar, kappa=FC.KAPPA)
