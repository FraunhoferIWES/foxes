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
    alpha
        The alpha parameter
    beta
        The beta parameter
    gamma
        The gamma parameter
    pre_rotor_only
        Calculate only the pre-rotor region
    induction
        The induction model


    """

    def __init__(
        self,
        superposition: str = "ws_linear",
        induction: str = "Madsen",
        alpha: float = 8 / 9,
        beta: float = np.sqrt(2),
        gamma: float = 1.1,
        pre_rotor_only: bool = False,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind speed superposition.
        induction
            The induction model
        alpha
            The alpha parameter
        beta
            The beta parameter
        gamma
            The gamma parameter
        pre_rotor_only
            Calculate only the pre-rotor region

        """
        super().__init__(wind_superposition=superposition)
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname}, gamma={self.gamma})"

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

    def _mu(self, x_R: np.ndarray) -> np.ndarray:
        """Helper function: define mu (eqn 11 from [1])"""
        return 1 + (x_R / np.sqrt(1 + x_R**2))

    def _a0(self, ct: np.ndarray, x_R: np.ndarray) -> np.ndarray:
        """Helper function: define a0 with gamma factor, eqn 8 from [2]"""
        induction = self.induction
        assert not isinstance(induction, str)
        return induction.ct2a(self.gamma * ct)

    def _a(self, ct: np.ndarray, x_R: np.ndarray) -> np.ndarray:
        """Helper function: define axial shape function (eqn 11 from [1])"""
        return self._a0(ct, x_R) * self._mu(x_R)

    def _r_half(self, x_R: np.ndarray) -> np.ndarray:
        """Helper function: using eqn 13 from [2]"""
        return np.sqrt(0.587 * (1.32 + x_R**2))

    def _rad_fn(self, x_R: np.ndarray, r_R: np.ndarray) -> np.ndarray:
        """Helper function: define radial shape function (eqn 12 from [1])"""
        with np.errstate(over="ignore"):
            result = (
                1 / np.cosh(self.beta * (r_R) / self._r_half(x_R))
            ) ** self.alpha  # * (x_R < 0)
        return result

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

        return None
