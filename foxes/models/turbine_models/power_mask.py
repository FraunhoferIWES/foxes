import numpy as np

from foxes.core import TurbineModel
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils import cubic_roots


class PowerMask(TurbineModel):
    """
    Invokes a maximal power value.

    This may correspond to turbine derating, if
    the maximal power value is below rated power.
    For higher values, a boost is introduced.

    The model updates the P and CT variables,
    so it is wise to use it after calling the
    turbine type model.

    Attributes
    ----------
    var_ws_P: str
        The wind speed variable for power lookup
    factor_P: float
        The power unit factor, e.g. 1000 for kW

    :group: models.turbine_models

    """

    def __init__(self, var_ws_P=FV.REWS3, factor_P=1.0e3):
        """
        Constructor.

        Parameters
        ----------
        var_ws_P: str
            The wind speed variable for power lookup
        factor_P: float
            The power unit factor, e.g. 1000 for kW

        """
        super().__init__()

        self.var_ws_P = var_ws_P
        self.factor_P = factor_P

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
        return [FV.P, FV.CT]

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(algo, verbosity)

        self._P_rated = []
        for t in algo.farm_controller.turbine_types:
            Pnom = FC.DTYPE(t.P_nominal)
            if np.isnan(Pnom):
                raise ValueError(
                    f"Model '{self.name}': P_nominal is NaN for turbine type '{t.name}'"
                )
            self._P_rated.append(Pnom)
        self._P_rated = np.array(self._P_rated, dtype=FC.DTYPE)

    @classmethod
    def update_P_ct(cls, data, max_P, rated_P, factor_P, var_ws=FV.REWS3, P_lim=100):
        # select power entries for which this is active:
        P = data[FV.P]
        sel = ~np.isnan(max_P) & (
            ((max_P < rated_P) & (P > max_P))
            | ((max_P > rated_P) & (P > rated_P - P_lim))
        )
        if np.any(sel):
            # apply selection:
            max_P = max_P[sel]
            ws = data[var_ws][sel]
            rho = data[FV.RHO][sel]
            r = data[FV.D][sel] / 2
            P = P[sel]
            ct = data[FV.CT][sel]
            ct[ct > 1.0] = 1.0

            # calculate power efficiency e of turbine
            # e is the ratio of the cp derived from the power curve
            # and the theoretical cp from the turbine induction
            cp = P / (0.5 * ws**3 * rho * np.pi * r**2) * factor_P
            a = 0.5 * (1 - np.sqrt(1 - ct))
            cp_a = 4 * a**3 - 8 * a**2 + 4 * a
            e = cp / cp_a
            del cp, a, cp_a, ct, P

            # calculating new cp for changed power
            cp = max_P / (0.5 * ws**3 * rho * np.pi * r**2) * factor_P

            # find roots:
            N = len(cp)
            a3 = np.full(N, 4.0, dtype=FC.DTYPE)
            a2 = np.full(N, -8.0, dtype=FC.DTYPE)
            a1 = np.full(N, 4.0, dtype=FC.DTYPE)
            a0 = -cp / e
            rts = cubic_roots(a0, a1, a2, a3)
            rts[np.isnan(rts)] = np.inf
            rts[rts <= 0.0] = np.inf
            a = np.min(rts, axis=1)
            del a0, a1, a2, a3, rts

            # set results:
            P = data[FV.P]
            ct = data[FV.CT]
            P[sel] = max_P
            ct[sel] = 4 * a * (1 - a)

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

        # prepare:
        max_P = fdata[FV.MAX_P]
        rated_P = self._P_rated[None, :]

        # calculate:
        self.update_P_ct(fdata, max_P, rated_P, self.factor_P, var_ws=self.var_ws_P)

        return {FV.P: fdata[FV.P], FV.CT: fdata[FV.CT]}
