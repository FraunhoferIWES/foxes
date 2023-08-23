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
        ws = np.zeros_like(heights)
        ws[:] = data[FV.WS]

        h0 = np.zeros_like(heights)
        h0[:] = data[FV.H]

        z0 = np.zeros_like(heights)
        z0[:] = data[FV.Z0]

        mol = np.zeros_like(heights)
        mol[:] = data[FV.MOL]

        out = np.zeros_like(heights)

        # neutral profiles:
        sel = np.isnan(mol) | (mol == 0.0)
        if np.any(sel):
            sws = ws[sel]
            sh0 = h0[sel]
            sz0 = z0[sel]
            sh = heights[sel]
            ustar = abl.neutral.ustar(sws, sh0, sz0, kappa=FC.KAPPA)
            out[sel] = abl.neutral.calc_ws(sh, sz0, ustar, kappa=FC.KAPPA)

        # stable profiles:
        sel = mol > 0.0
        if np.any(sel):
            sws = ws[sel]
            sh0 = h0[sel]
            sz0 = z0[sel]
            smo = mol[sel]
            sh = heights[sel]
            ustar = abl.stable.ustar(sws, sh0, sz0, smo, kappa=FC.KAPPA)
            psi = abl.stable.psi(sh, smo)
            out[sel] = abl.stable.calc_ws(sh, sz0, ustar, psi, kappa=FC.KAPPA)

        # unstable profiles:
        sel = mol < 0.0
        if np.any(sel):
            sws = ws[sel]
            sh0 = h0[sel]
            sz0 = z0[sel]
            smo = mol[sel]
            sh = heights[sel]
            ustar = abl.unstable.ustar(sws, sh0, sz0, smo, kappa=FC.KAPPA)
            psi = abl.unstable.psi(sh, smo)
            out[sel] = abl.unstable.calc_ws(sh, sz0, ustar, psi, kappa=FC.KAPPA)

        return out
