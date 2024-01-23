import numpy as np

from foxes.core import VerticalProfile
from foxes.utils import abl
import foxes.constants as FC
import foxes.variables as FV


class ABLLogWsProfile(VerticalProfile):
    """
    The neutral/stable/unstable ABL wind speed log profile.

    This profile picks the profile according to the mol value
    (neutral: mol = None or mol = 0)

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
            return [FV.USTAR, FV.Z0, FV.MOL]
        else:
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

        z0 = np.zeros_like(heights)
        z0[:] = data[FV.Z0]

        mol = np.zeros_like(heights)
        mol[:] = data[FV.MOL]

        if self.ustar_input:
            ustar = data[FV.USTAR]
        else:
            ws = np.zeros_like(heights)
            ws[:] = data[FV.WS]

            h0 = np.zeros_like(heights)
            h0[:] = data[FV.H]

        out = np.zeros_like(heights)

        # neutral profiles:
        sel = np.isnan(mol) | (mol == 0.0)
        if np.any(sel):
            sz0 = z0[sel]
            sh = heights[sel]
            if self.ustar_input:
                sus = ustar[sel]
            else:
                sus = abl.neutral.ustar(ws[sel], h0[sel], sz0, kappa=FC.KAPPA)
            out[sel] = abl.neutral.calc_ws(sh, sz0, sus, kappa=FC.KAPPA)

        # stable profiles:
        sel = mol > 0.0
        if np.any(sel):
            sz0 = z0[sel]
            smo = mol[sel]
            sh = heights[sel]
            if self.ustar_input:
                sus = ustar[sel]
            else:
                sus = abl.stable.ustar(ws[sel], h0[sel], sz0, smo, kappa=FC.KAPPA)
            psi = abl.stable.psi(sh, smo)
            out[sel] = abl.stable.calc_ws(sh, sz0, sus, psi, kappa=FC.KAPPA)

        # unstable profiles:
        sel = mol < 0.0
        if np.any(sel):
            sz0 = z0[sel]
            smo = mol[sel]
            sh = heights[sel]
            if self.ustar_input:
                sus = ustar[sel]
            else:
                sus = abl.unstable.ustar(ws[sel], h0[sel], sz0, smo, kappa=FC.KAPPA)
            psi = abl.unstable.psi(sh, smo)
            out[sel] = abl.unstable.calc_ws(sh, sz0, sus, psi, kappa=FC.KAPPA)

        return out
