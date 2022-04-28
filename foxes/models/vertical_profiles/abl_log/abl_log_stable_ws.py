
from foxes.core import VerticalProfile
from foxes.tools.abl import stable
import foxes.constants as FC
import foxes.variables as FV

class ABLLogStableWsProfile(VerticalProfile):
    """
    The stable ABL wind speed log profile.
    """

    def input_vars(self):
        return [FV.WS, FV.H, FV.Z0, FV.MOL]

    def calculate(self, data, heights):

        ws  = data[FV.WS]
        h0  = data[FV.H]
        z0  = data[FV.Z0]
        mol = data[FV.MOL]

        ustar = stable.ustar(ws, h0, z0, mol, kappa=FC.KAPPA)
        psi   = stable.psi(heights, mol)

        return stable.calc_ws(heights, z0, ustar, psi, kappa=FC.KAPPA)
