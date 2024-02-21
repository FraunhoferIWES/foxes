import numpy as np

from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC


class SelfSimilar(TurbineInductionModel):
    """
    The self-similar induction wake model
    from Troldborg and Meyer Forsting

    The individual wake effects are superposed linearly,
    without invoking a wake superposition model.

    Notes
    -----
    References:
    [1] Troldborg, Niels, and Alexander Raul Meyer Forsting.
    "A simple model of the wind turbine induction zone derived
    from numerical simulations."
    Wind Energy 20.12 (2017): 2011-2020.
    https://onlinelibrary.wiley.com/doi/full/10.1002/we.2137

    [2] Forsting, Alexander R. Meyer, et al.
    "On the accuracy of predicting wind-farm blockage."
    Renewable Energy (2023).
    https://www.sciencedirect.com/science/article/pii/S0960148123007620

    Attributes
    ----------
    pre_rotor_only: bool
        Calculate only the pre-rotor region
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(self, pre_rotor_only=False, induction="Madsen"):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor_only: bool
            Calculate only the pre-rotor region
        induction: foxes.core.AxialInductionModel or str
            The induction model

        """
        super().__init__()
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.induction]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

    def init_wake_deltas(self, algo, mdata, fdata, pdata, wake_deltas):
        """
        Initialize wake delta storage.

        They are added on the fly to the wake_deltas dict.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        wake_deltas: dict
            The wake deltas storage, add wake deltas
            on the fly. Keys: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        n_states = mdata.n_states
        n_points = pdata.n_points
        wake_deltas[FV.WS] = np.zeros((n_states, n_points), dtype=FC.DTYPE)

    def _mu(self, x_R):
        """Helper function: define mu (eqn 11 from [1])"""
        return 1 + (x_R / np.sqrt(1 + x_R**2))

    def _a0(self, ct, x_R, gamma=1.1):
        """Helper function: define a0 with gamma factor, eqn 8 from [2]"""
        return self.induction.ct2a(gamma * ct)

    def _a(self, ct, x_R):
        """Helper function: define axial shape function (eqn 11 from [1])"""
        return self._a0(ct, x_R) * self._mu(x_R)

    def _r_half(self, x_R):
        """Helper function: using eqn 13 from [2]"""
        return np.sqrt(0.587 * (1.32 + x_R**2))

    def _rad_fn(self, x_R, r_R, beta=np.sqrt(2), alpha=8 / 9):
        """Helper function: define radial shape function (eqn 12 from [1])"""
        return (1 / np.cosh(beta * (r_R) / self._r_half(x_R))) ** alpha  # * (x_R < 0)

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_coos,
        wake_deltas,
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """

        # get ct
        ct = self.get_data(
            FV.CT,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # get ws
        ws = self.get_data(
            FV.REWS,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_POINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            pdata=pdata,
            upcast=True,
            states_source_turbine=states_source_turbine,
        )

        # get x, r and R etc
        x = wake_coos[:, :, 0]
        y = wake_coos[:, :, 1]
        z = wake_coos[:, :, 2]
        R = D / 2
        x_R = x / R
        r = np.sqrt(y**2 + z**2)
        r_R = r / R

        # select values
        sp_sel = (ct > 0) & (x < 0)  # upstream
        if np.any(sp_sel):
            # velocity eqn 10 from [1]
            xr = x_R[sp_sel]
            blockage = (
                ws[sp_sel] * self._a(ct[sp_sel], xr) * self._rad_fn(xr, r_R[sp_sel])
            )
            wake_deltas[FV.WS][sp_sel] -= blockage

        # set area behind to mirrored value EXCEPT for area behind turbine
        if not self.pre_rotor_only:
            sp_sel = (ct > 0) & (x > 0) & (r_R > 1)
            if np.any(sp_sel):
                # velocity eqn 10 from [1]
                xr = x_R[sp_sel]
                blockage = (
                    ws[sp_sel]
                    * self._a(ct[sp_sel], -xr)
                    * self._rad_fn(-xr, r_R[sp_sel])
                )
                wake_deltas[FV.WS][sp_sel] += blockage

        return wake_deltas
