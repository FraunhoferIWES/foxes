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
        superposition: str = "ws_linear",
        induction: str = "Madsen",
        pre_rotor_only: bool = False,
    ) -> None:
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
        dws: bool
            If True, this model affects wind speed

        """
        return True

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
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
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
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
        R_sel = D[sp_sel] / 2
        xi = r_sph_sel / R_sel

        def add_wake(
            sp_sel: np.ndarray,
            wake_deltas: dict[str, np.ndarray],
            blockage: np.ndarray,
        ) -> None:
            """adds to wake deltas"""
            if self.has_vector_wind_superp:
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

        if np.any(sp_sel):
            induction = self.induction
            assert not isinstance(induction, str)
            blockage = induction.ct2a(ct_sel) * (1 + -xi / np.sqrt(1 + xi**2))
            add_wake(sp_sel, wake_deltas, -blockage)

        if not self.pre_rotor_only:
            sp_sel = (
                (ct > 1e-8) & (x > 0) & (r > D / 2)
            )  # mirror in rotor plane and inverse blockage, but not directly behind rotor
            ct_sel = ct[sp_sel]
            r_sph_sel = r_sph[sp_sel]
            R_sel = D[sp_sel] / 2
            xi = r_sph_sel / R_sel
            if np.any(sp_sel):
                induction = self.induction
                assert not isinstance(induction, str)
                blockage = induction.ct2a(ct_sel) * (1 + -xi / np.sqrt(1 + xi**2))
                add_wake(sp_sel, wake_deltas, blockage)
        return None
