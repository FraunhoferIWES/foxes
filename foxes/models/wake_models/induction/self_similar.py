import numpy as np

from foxes.config import config
from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC


class SelfSimilar(TurbineInductionModel):
    """
    The self-similar induction wake model
    from Troldborg and Meyer Forsting

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
    alpha: float
        The alpha parameter
    beta: float
        The beta parameter
    gamma: float
        The gamma parameter
    pre_rotor_only: bool
        Calculate only the pre-rotor region
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(
        self,
        superposition="ws_linear",
        induction="Madsen",
        alpha=8 / 9,
        beta=np.sqrt(2),
        gamma=1.1,
        pre_rotor_only=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind speed superposition.
        induction: foxes.core.AxialInductionModel or str
            The induction model
        alpha: float
            The alpha parameter
        beta: float
            The beta parameter
        gamma: float
            The gamma parameter
        pre_rotor_only: bool
            Calculate only the pre-rotor region

        """
        super().__init__(wind_superposition=superposition)
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname}, gamma={self.gamma})"

    @property
    def affects_ws(self):
        """
        Flag for wind speed wake models

        Returns
        -------
        dws: bool
            If True, this model affects wind speed

        """
        return True

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return super().sub_models() + [self.induction]

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
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        if self.has_uv:
            duv = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints, 2),
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}

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

    def _rad_fn(self, x_R, r_R):
        """Helper function: define radial shape function (eqn 12 from [1])"""
        with np.errstate(over="ignore"):
            result = (
                1 / np.cosh(self.beta * (r_R) / self._r_half(x_R))
            ) ** self.alpha  # * (x_R < 0)
        return result

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
            in the downwind order
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

        def add_wake(sp_sel, wake_deltas, blockage):
            """adds to wake deltas"""
            if self.has_uv:
                assert self.has_vector_wind_superp, (
                    f"Wake model {self.name}: Missing vector wind superposition, got '{self.wind_superposition}'"
                )
                wdeltas = {FV.WS: blockage}
                self.vec_superp.wdeltas_ws2uv(
                    algo, fdata, tdata, downwind_index, wdeltas, sp_sel
                )
                wake_deltas[FV.UV] = self.vec_superp.add_wake_vector(
                    algo,
                    mdata,
                    fdata,
                    tdata,
                    downwind_index,
                    sp_sel,
                    wake_deltas[FV.UV],
                    wdeltas.pop(FV.UV),
                )
            else:
                self.superp[FV.WS].add_wake(
                    algo,
                    mdata,
                    fdata,
                    tdata,
                    downwind_index,
                    sp_sel,
                    FV.WS,
                    wake_deltas[FV.WS],
                    blockage,
                )

        # select values
        sp_sel = (ct > 1e-8) & (x_R <= 0)  # upstream
        if np.any(sp_sel):
            # velocity eqn 10 from [1]
            xr = x_R[sp_sel]
            blockage = self._a(ct[sp_sel], xr) * self._rad_fn(xr, r_R[sp_sel])

            add_wake(sp_sel, wake_deltas, -blockage)

        # set area behind to mirrored value EXCEPT for area behind turbine
        if not self.pre_rotor_only:
            sp_sel = (ct > 1e-8) & (x_R > 0) & (r_R > 1)
            if np.any(sp_sel):
                # velocity eqn 10 from [1]
                xr = x_R[sp_sel]
                blockage = self._a(ct[sp_sel], -xr) * self._rad_fn(-xr, r_R[sp_sel])

                add_wake(sp_sel, wake_deltas, blockage)

        return wake_deltas
