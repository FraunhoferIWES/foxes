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

    def __init__(self, pre_rotor_only=False, induction="Madsen", gamma=1.1):
        """
        Constructor.

        Parameters
        ----------
        pre_rotor_only: bool
            Calculate only the pre-rotor region
        induction: foxes.core.AxialInductionModel or str
            The induction model
        gamma: float, default=1.1
            The parameter that multiplies Ct in the ct2a calculation

        """
        super().__init__()
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only
        self.gamma = gamma

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}(gamma={self.gamma}, induction={iname})"

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

    def new_wake_deltas(self, algo, mdata, fdata, tdata):
        """
        Creates new empty wake delta arrays.

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
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return {FV.WS: np.zeros_like(tdata[FC.TARGETS][..., 0])}

    def _mu(self, x_R):
        """Helper function: define mu (eqn 11 from [1])"""
        return 1 + (x_R / np.sqrt(1 + x_R**2))

    def _a0(self, ct, x_R):
        """Helper function: define a0 with gamma factor, eqn 8 from [2]"""
        return self.induction.ct2a(self.gamma * ct)

    def _a(self, ct, x_R):
        """Helper function: define axial shape function (eqn 11 from [1])"""
        return self._a0(ct, x_R) * self._mu(x_R)

    def _r_half(self, x_R):
        """Helper function: using eqn 13 from [2]"""
        return np.sqrt(0.587 * (1.32 + x_R**2))

    def _rad_fn(self, x_R, r_R, beta=np.sqrt(2), alpha=8 / 9):
        """Helper function: define radial shape function (eqn 12 from [1])"""
        return (1 / np.cosh(beta * (r_R) / self._r_half(x_R))) ** alpha  # * (x_R < 0)

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        # get ct
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
        )

        # get ws
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
        )

        # get R
        R = 0.5 * self.get_data(
            FV.D,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=False,
            downwind_index=downwind_index,
        )

        # get x, r and R etc. Rounding for safe x < 0 condition below
        x_R = np.round(wake_coos[..., 0] / R, 12)
        r_R = np.linalg.norm(wake_coos[..., 1:3], axis=-1) / R

        # select values
        sp_sel = (ct > 0) & (x_R <= 0)  # upstream
        if np.any(sp_sel):
            # velocity eqn 10 from [1]
            xr = x_R[sp_sel]
            blockage = (
                ws[sp_sel] * self._a(ct[sp_sel], xr) * self._rad_fn(xr, r_R[sp_sel])
            )
            wake_deltas[FV.WS][sp_sel] -= blockage

        # set area behind to mirrored value EXCEPT for area behind turbine
        if not self.pre_rotor_only:
            sp_sel = (ct > 0) & (x_R > 0) & (r_R > 1)
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
