from foxes.core import PointDataModel
from foxes.utils import ustar2ti
import foxes.variables as FV


class Ustar2TI(PointDataModel):
    """
    Calculates TI from Ustar, using TI = Ustar / (kappa*WS)

    Attributes
    ----------
    max_ti: float
        Upper limit of the computed TI values

    :group: models.point_models

    """

    def __init__(self, max_ti=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        max_ti: float, optional
            Upper limit of the computed TI values
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.max_ti = max_ti

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return [FV.TI]

    def calculate(self, algo, mdata, fdata, tdata):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        ustar = tdata[FV.USTAR]
        ws = tdata[FV.WS]

        ti = ustar2ti(ustar, ws, self.max_ti)

        return {FV.TI: ti}
