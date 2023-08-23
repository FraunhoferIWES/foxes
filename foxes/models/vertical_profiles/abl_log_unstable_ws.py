from foxes.core import VerticalProfile
from foxes.utils.abl import unstable
import foxes.constants as FC
import foxes.variables as FV


class ABLLogUnstableWsProfile(VerticalProfile):
    """
    The unstable ABL wind speed log profile.

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
        return [FV.WS, FV.H, FV.Z0, FV.MOL]

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
        ws = data[FV.WS]
        h0 = data[FV.H]
        z0 = data[FV.Z0]
        mol = data[FV.MOL]

        ustar = unstable.ustar(ws, h0, z0, mol, kappa=FC.KAPPA)
        psi = unstable.psi(heights, mol)

        return unstable.calc_ws(heights, z0, ustar, psi, kappa=FC.KAPPA)
