import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC


class YAWM2YAW(TurbineModel):
    """
    Calculates absolute yaw (i.e. YAWM) from delta
    yaw (i.e. YAWM)

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
        return [FV.YAW]

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
        yawm = self.get_data(
            FV.YAWM, FC.STATE_TURBINE, lookup="f", fdata=fdata, upcast=True
        )[st_sel]
        wd = self.get_data(
            FV.WD, FC.STATE_TURBINE, lookup="f", fdata=fdata, upcast=True
        )[st_sel]

        yaw = fdata[FV.YAW]
        yaw[st_sel] = np.mod(wd + yawm, 360.0)

        return {FV.YAW: yaw}
