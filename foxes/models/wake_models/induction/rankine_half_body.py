from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.config import config
from foxes.models.wake_models.turbine_induction_model import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model


class RankineHalfBody(TurbineInductionModel):
    """
    The Rankine half body induction wake model

    Notes
    -----
    Reference:
    B Gribben and G Hawkes
    "A potential flow model for wind turbine induction and wind farm blockage"
    Techincal Paper, Frazer-Nash Consultancy, 2019
    https://www.fnc.co.uk/media/o5eosxas/a-potential-flow-model-for-wind-turbine-induction-and-wind-farm-blockage.pdf

    Attributes
    ----------
    induction
        The induction model


    """

    def __init__(
        self, superposition: str = "vector", induction: str = "Madsen"
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind speed deficit superposition.
        induction
            The induction model

        """
        super().__init__(wind_superposition=superposition, other_superpositions={})
        self.induction = induction

        self._has_uv = True

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname})"

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

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
        duv: np.ndarray = np.zeros(
            (n_states, n_targets, n_tpoints, 2),
            dtype=config.dtype_double,
        )
        return {FV.UV: duv}

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
            downwind_index=downwind_index,
            upcast=False,
        )

        # get ws:
        ws = self.get_data(
            FV.REWS,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # calc m (page 7, skipping pi everywhere)
        induction = self.induction
        assert not isinstance(induction, str)
        m = 2 * ws * induction.ct2a(ct) * (D / 2) ** 2

        # get r and theta
        x = np.round(wake_coos[..., 0], 12)
        r = np.linalg.norm(wake_coos[..., 1:], axis=-1)
        r_sph = np.sqrt(r**2 + x**2)
        theta = np.arctan2(r, x)

        # define rankine half body shape (page 3)
        RHB_shape = (
            np.cos(theta) - (2 / (m + 1e-15)) * ws * (r_sph * np.sin(theta)) ** 2
        )

        # stagnation point condition
        xs = -np.sqrt(m / (4 * ws + 1e-15))

        # set values out of body shape
        st_sel = (ct > 1e-8) & ((RHB_shape < -1) | (x < xs))
        if np.any(st_sel):
            # apply selection
            xyz = wake_coos[st_sel]

            # calc velocity components
            vel_factor = m[st_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas[FV.UV][st_sel] += vel_factor[:, None] * xyz[:, :2]

        # set values inside body shape
        st_sel = (ct > 1e-8) & (RHB_shape >= -1) & (x >= xs) & (x <= 0)
        if np.any(st_sel):
            # apply selection
            xyz = np.zeros_like(wake_coos[st_sel])
            xyz[:, 0] = xs[st_sel]

            # calc velocity components
            vel_factor = m[st_sel] / (4 * np.linalg.norm(xyz, axis=-1) ** 3)
            wake_deltas[FV.UV][st_sel, 0] += vel_factor * xyz[:, 0]
