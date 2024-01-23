from foxes.core import VerticalProfile
from foxes.utils.abl import neutral
import foxes.constants as FC
import foxes.variables as FV


class ABLLogNeutralWsProfile(VerticalProfile):
    """
    The neutral ABL wind speed log profile.

    Attributes
    ----------
    ustar_input: bool
        Flag for using ustar as an input

    :group: models.vertical_profiles

    """

    def __init__(self, *args, ustar_input=False, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional arguments for VerticalProfile
        ustar_input: bool
            Flag for using ustar as an input
        kwargs: dict, optional
            Additional arguments for VerticalProfile

        """
        super().__init__(*args, **kwargs)
        self.ustar_input = ustar_input

    def input_vars(self):
        """
        The input variables needed for the profile
        calculation.

        Returns
        -------
        vars: list of str
            The variable names

        """
        if self.ustar_input:
            return [FV.USTAR, FV.Z0]
        else:
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
        if self.ustar_input:
            ustar = data[FV.USTAR]
        else:
            h0 = data[FV.H]
            ws = data[FV.WS]
            ustar = neutral.ustar(ws, h0, z0, kappa=FC.KAPPA)

        return neutral.calc_ws(heights, z0, ustar, kappa=FC.KAPPA)
