import numpy as np

from foxes.core import TurbineOrder
from foxes.tools import wd2uv
import foxes.variables as FV

class OrderWD(TurbineOrder):
    """
    Order the turbines by projecting the coordinates
    onto the normalized mean wind direction vector.

    Parameters
    ----------
    var_wd : str
        The wind direction variable
    
    Attributes
    ----------
    var_wd : str
        The wind direction variable

    """

    def __init__(self, var_wd=FV.WD):
        super().__init__()
        self.var_wd = var_wd

    def calculate(self, algo, mdata, fdata):
        """"
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        
        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        n  = np.mean(wd2uv(fdata[self.var_wd], axis=1), axis=-1)
        xy = fdata[FV.TXYH][:, :, :2]

        order = np.argsort(np.einsum('std,sd->st', xy, n), axis=-1)

        return {FV.ORDER: order}
