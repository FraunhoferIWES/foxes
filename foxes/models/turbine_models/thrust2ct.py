import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV


class Thrust2Ct(TurbineModel):
    """
    Calculates ct from thrust force data.

    Attributes
    ----------
    thrust_var: str
        Name of the thrust variable
    WSCT: str
        The wind speed variable for ct lookup

    :group: models.turbine_models

    """

    def __init__(self, thrust_var=FV.T, var_ws_ct=FV.REWS2):
        """
        Constructor.

        Parameters
        ----------
        thrust_var: str
            Name of the thrust variable
        var_ws_ct: str
            The wind speed variable for ct lookup

        """
        super().__init__()
        self.thrust_var = thrust_var
        self.WSCT = var_ws_ct

    def __repr__(self):
        a = f"thrust_var={self.thrust_var}, var_ws_ct={self.WSCT}"
        return f"{type(self).__name__}({a})"

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
        return [FV.CT]

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        ct = fdata[FV.CT]

        T = fdata[self.thrust_var][st_sel]
        rho = fdata[FV.RHO][st_sel]
        A = np.pi * (fdata[FV.D][st_sel] / 2) ** 2
        ws = fdata[self.WSCT][st_sel]

        ct[st_sel] = 2 * T / (rho * A * ws**2)

        return {FV.CT: ct}
