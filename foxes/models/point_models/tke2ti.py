import numpy as np

from foxes.core import PointDataModel
import foxes.variables as FV


class TKE2TI(PointDataModel):
    """
    Calculates TI from TKE, using TI = sqrt( 3/2 * TKE) / WS

    :group: models.point_models

    """

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

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
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
        tke = pdata[FV.TKE]
        ws = pdata[FV.WS]

        return {FV.TI: np.sqrt(1.5 * tke) / ws}
