import numpy as np

from foxes.config import config
from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC


class Rathmann(TurbineInductionModel):
    """
    The Rathmann induction wake model

    The individual wake effects are superposed linearly,
    without invoking a wake superposition model.

    Notes
    -----
    Reference:
    Forsting, Alexander R. Meyer, et al.
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

    def __init__(
        self,
        superposition="ws_linear",
        induction="Madsen",
        pre_rotor_only=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind speed superposition
        induction: foxes.core.AxialInductionModel or str
            The induction model
        pre_rotor_only: bool
            Calculate only the pre-rotor region

        """
        super().__init__(wind_superposition=superposition)
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname})"

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
        # get ct:
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

        # get D
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

        def mu(x_R):
            """axial shape function at r=0 from vortex cylinder model (eqn 11)"""
            return 1 + x_R / (np.sqrt(1 + x_R**2))

        def G(x_R, r_R):
            """radial shape function eqn 20"""
            sin_2_alpha = (2 * x_R) / np.sqrt(
                (x_R**2 + (r_R - 1) ** 2) * (x_R**2 + (r_R + 1) ** 2)
            )  # eqn 19
            sin_alpha = np.sqrt(
                0.5 * (1 - np.sqrt(1 - sin_2_alpha**2))
            )  # derived from cos(2a)**2 + sin(2a)**2 = 1
            sin_beta = 1 / np.sqrt(x_R**2 + r_R**2 + 1)  # eqn 19
            return sin_alpha * sin_beta * (1 + x_R**2)

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

        # ws delta in front of rotor
        sp_sel = (ct > 1e-8) & (x_R <= 0)
        if np.any(sp_sel):
            xr = x_R[sp_sel]
            a = self.induction.ct2a(ct[sp_sel])
            blockage = a * mu(xr) * G(xr, r_R[sp_sel])  # eqn 10

            add_wake(sp_sel, wake_deltas, -blockage)

        # ws delta behind rotor
        if not self.pre_rotor_only:
            # mirror -blockage in rotor plane
            sp_sel = (ct > 1e-8) & (x_R > 0) & (r_R > 1)
            if np.any(sp_sel):
                xr = x_R[sp_sel]
                a = self.induction.ct2a(ct[sp_sel])
                blockage = a * mu(-xr) * G(-xr, r_R[sp_sel])  # eqn 10

                add_wake(sp_sel, wake_deltas, blockage)

        return wake_deltas
