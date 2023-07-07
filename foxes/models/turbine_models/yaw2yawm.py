import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils import delta_wd


class YAW2YAWM(TurbineModel):
    """
    Calculates delta yaw (i.e. YAWM) from absolute
    yaw (i.e. YAW)

    :group: models.turbine_models

    """

    def output_farm_vars(self, algo):
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
        return [FV.YAWM]

    def calculate(self, algo, mdata, fdata, st_sel):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        st_sel: numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        yaw = self.get_data(
            FV.YAW, FC.STATE_TURBINE, lookup="f", fdata=fdata, upcast=True
        )[st_sel]
        wd = self.get_data(
            FV.WD, FC.STATE_TURBINE, lookup="f", fdata=fdata, upcast=True
        )[st_sel]

        yawm = fdata[FV.YAWM]
        yawm[st_sel] = delta_wd(wd, yaw)

        return {FV.YAWM: yawm}
