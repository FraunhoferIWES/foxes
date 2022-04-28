import numpy as np

from foxes.core import TurbineOrder
from foxes.tools import wd2uv
import foxes.variables as FV

class OrderWD(TurbineOrder):

    def __init__(self, var_wd=FV.WD):
        super().__init__()
        self.var_wd = var_wd

    def calculate(self, algo, fdata):

        n  = np.mean(wd2uv(fdata[self.var_wd], axis=1), axis=-1)
        xy = fdata[FV.TXYH][:, :, :2]

        order = np.argsort(np.einsum('std,sd->st', xy, n), axis=-1)

        return {FV.ORDER: order}
