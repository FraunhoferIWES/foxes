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
    P_lim: float
        Threshold power delta for boosts
    induction: foxes.core.AxialInductionModel
        The induction model

    :group: models.turbine_models

    """

    def __init__(self, var_ws_P=FV.REWS3, factor_P=1.0e3, P_lim=100, induction="Betz"):
        """
        Constructor.

        Parameters
        ----------
        var_ws_P: str
            The wind speed variable for power lookup
        factor_P: float
            The power unit factor, e.g. 1000 for kW
        P_lim: float
            Threshold power delta for boosts
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__()

        self.var_ws_P = var_ws_P
        self.factor_P = factor_P
        self.P_lim = P_lim
        self.induction = induction

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        a = f"var_ws_P={self.var_ws_P}, P_lim={self.P_lim}, induction={iname}"
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
        return [FV.P, FV.CT]

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.induction]

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
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
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
        # prepare:
        P = fdata[FV.P]
        max_P = fdata[FV.MAX_P]
        P_rated = self._P_rated[None, :]

        # select power entries for which this is active:
        sel = np.zeros((fdata.n_states, fdata.n_turbines), dtype=bool)
        sel[st_sel] = True
        sel = (
            sel
            & ~np.isnan(max_P)
            & (
                ((max_P < P_rated) & (P > max_P))
                | ((max_P > P_rated) & (P > P_rated - self.P_lim))
            )
        )
        if np.any(sel):
            # apply selection:
            max_P = max_P[sel]
            ws = fdata[self.var_ws_P][sel]
            rho = fdata[FV.RHO][sel]
            r = fdata[FV.D][sel] / 2
            P = P[sel]
            ct = fdata[FV.CT][sel]

            # calculate power efficiency e of turbine
            # e is the ratio of the cp derived from the power curve
            # and the theoretical cp from the turbine induction
            cp = P / (0.5 * ws**3 * rho * np.pi * r**2) * self.factor_P
            a = self.induction.ct2a(ct)
            cp_a = 4 * a**3 - 8 * a**2 + 4 * a
            e = cp / cp_a
            del cp, a, cp_a, ct, P

            # calculating new cp for changed power
            cp = max_P / (0.5 * ws**3 * rho * np.pi * r**2) * self.factor_P

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
            P = fdata[FV.P]
            ct = fdata[FV.CT]
            P[sel] = max_P
            ct[sel] = 4 * a * (1 - a)

        return {FV.P: fdata[FV.P], FV.CT: fdata[FV.CT]}
