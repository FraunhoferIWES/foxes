from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.config import config
from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model


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
    pre_rotor_only
        Calculate only the pre-rotor region
    induction
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(
        self,
        superposition: str = "ws_linear",
        induction: str = "Madsen",
        pre_rotor_only: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind speed superposition
        induction
            The induction model
        pre_rotor_only
            Calculate only the pre-rotor region

        """
        super().__init__(wind_superposition=superposition)
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname})"

    @property
    def affects_ws(self) -> bool:
        """
        Flag for wind speed wake models

        Returns
        -------
        dws
            If True, this model affects wind speed

        """
        return True

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        smdls = super().sub_models()
        if not isinstance(self.induction, str):
            smdls.append(self.induction)
        return smdls

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def new_wake_deltas(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
        """
        Creates new empty wake delta arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data

        Returns
        -------
        wake_deltas
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        assert n_states is not None and n_targets is not None and n_tpoints is not None
        if self.has_uv:
            duv: np.ndarray = np.zeros(
                (n_states, n_targets, n_tpoints, 2),
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws: np.ndarray = np.zeros(
                (n_states, n_targets, n_tpoints),
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}

    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_coos: np.ndarray,
        wake_deltas: dict[str, np.ndarray],
    ) -> None:
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wake_coos
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas
            The wake deltas. Key: variable name,
            value
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

        def mu(x_R: np.ndarray) -> np.ndarray:
            """axial shape function at r=0 from vortex cylinder model (eqn 11)"""
            return 1 + x_R / (np.sqrt(1 + x_R**2))

        def G(x_R: np.ndarray, r_R: np.ndarray) -> np.ndarray:
            """radial shape function eqn 20"""
            sin_2_alpha = (2 * x_R) / np.sqrt(
                (x_R**2 + (r_R - 1) ** 2) * (x_R**2 + (r_R + 1) ** 2)
            )  # eqn 19
            sin_alpha = np.sqrt(
                0.5 * (1 - np.sqrt(1 - sin_2_alpha**2))
            )  # derived from cos(2a)**2 + sin(2a)**2 = 1
            sin_beta = 1 / np.sqrt(x_R**2 + r_R**2 + 1)  # eqn 19
            return sin_alpha * sin_beta * (1 + x_R**2)

        def add_wake(
            sp_sel: np.ndarray,
            wake_deltas: dict[str, np.ndarray],
            blockage: np.ndarray,
        ) -> None:
            """adds to wake deltas"""
            if self.has_uv:
                assert self.has_vector_wind_superp, (
                    f"Wake model {self.name}: Missing vector wind superposition, got '{self.wind_superposition}'"
                )
                vec_superp = self.vec_superp
                assert vec_superp is not None
                wdeltas = {FV.WS: blockage}
                vec_superp.wdeltas_ws2uv(
                    algo, fdata, tdata, downwind_index, wdeltas, sp_sel
                )
                wake_deltas[FV.UV] = vec_superp.add_wake_vector(
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
            induction = self.induction
            assert not isinstance(induction, str)
            a = induction.ct2a(ct[sp_sel])
            blockage = a * mu(xr) * G(xr, r_R[sp_sel])  # eqn 10

            add_wake(sp_sel, wake_deltas, -blockage)

        # ws delta behind rotor
        if not self.pre_rotor_only:
            # mirror -blockage in rotor plane
            sp_sel = (ct > 1e-8) & (x_R > 0) & (r_R > 1)
            if np.any(sp_sel):
                xr = x_R[sp_sel]
                induction = self.induction
                assert not isinstance(induction, str)
                a = induction.ct2a(ct[sp_sel])
                blockage = a * mu(-xr) * G(-xr, r_R[sp_sel])  # eqn 10

                add_wake(sp_sel, wake_deltas, blockage)

        return None
