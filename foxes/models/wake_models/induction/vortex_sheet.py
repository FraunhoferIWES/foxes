import numpy as np

from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC


class VortexSheet(TurbineInductionModel):
    """
    The Vortex Sheet model implemented with a radial dependency

    Notes
    -----
    Reference:
    Medici, D., et al. "The upstream flow of a wind turbine: blockage effect." Wind Energy 14.5 (2011): 691-697.
    https://doi.org/10.1002/we.451

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
        super().__init__()
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only
        self._superp_name = superposition

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self._superp_name}, induction={iname})"

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self._superp, self.induction]

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
        self._superp = algo.mbook.wake_superpositions[self._superp_name]
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

    def new_wake_deltas(self, algo, mdata, fdata, tdata):
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
        return {FV.WS: np.zeros_like(tdata[FC.TARGETS][..., 0])}

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
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
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

        # get x, y and z. Rounding for safe x < 0 condition
        x = np.round(wake_coos[..., 0], 12)
        y = wake_coos[..., 1]
        z = wake_coos[..., 2]
        r = np.sqrt(y**2 + z**2)
        r_sph = np.sqrt(r**2 + x**2)

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

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
        )

        sp_sel = (ct > 1e-8) & (x <= 0)
        ct_sel = ct[sp_sel]
        r_sph_sel = r_sph[sp_sel]
        D_sel = D[sp_sel]

        if np.any(sp_sel):
            blockage = self.induction.ct2a(ct_sel) * (
                (1 + 2 * r_sph_sel / D_sel) * (1 + (2 * r_sph_sel / D_sel) ** 2)
            ) ** (-0.5)

            self._superp.add_wake(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                sp_sel,
                FV.WS,
                wake_deltas[FV.WS],
                -blockage,
            )

        if not self.pre_rotor_only:
            sp_sel = (
                (ct > 1e-8) & (x > 0) & (r > D / 2)
            )  # mirror in rotor plane and inverse blockage, but not directly behind rotor
            ct_sel = ct[sp_sel]
            r_sph_sel = r_sph[sp_sel]
            D_sel = D[sp_sel]
            if np.any(sp_sel):
                blockage = self.induction.ct2a(ct_sel) * (
                    (1 + 2 * r_sph_sel / D_sel) * (1 + (2 * r_sph_sel / D_sel) ** 2)
                ) ** (-0.5)

                self._superp.add_wake(
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

        return wake_deltas

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        amb_results,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)
        wake_deltas: dict
            The wake deltas object at the selected target
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_targets, n_tpoints)

        """
        wake_deltas[FV.WS] = self._superp.calc_final_wake_delta(
            algo, mdata, fdata, FV.WS, amb_results[FV.WS], wake_deltas[FV.WS]
        )
