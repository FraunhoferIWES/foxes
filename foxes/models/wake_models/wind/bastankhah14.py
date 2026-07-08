from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import WakeK
from foxes.models.wake_models.gaussian import GaussianWakeModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model


class Bastankhah2014(GaussianWakeModel):
    """
    The Bastankhah 2014 wake model

    Notes
    -----
    Reference:
    "A new analytical model for wind-turbine wakes"
    Majid Bastankhah, Fernando Porté-Agel
    https://doi.org/10.1016/j.renene.2014.01.002

    Attributes
    ----------
    sbeta_factor: float
        Factor multiplying sbeta, only relevant if sbeta is not set
    sbeta: float
        If set, sbeta is fixed to this value, otherwise it
        is calculated from axial induction
    induction: foxes.core.AxialInductionModel
        The axial induction model
    wake_k: foxes.core.WakeK
        Handler for the wake growth parameter k

    :group: models.wake_models.wind

    """

    def __init__(
        self,
        superposition: str,
        sbeta_factor: float = 0.2,
        sbeta: float | None = None,
        induction: str = "Madsen",
        **wake_k: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind speed deficit superposition.
        sbeta_factor: float
            Factor multiplying sbeta, only relevant if sbeta is not set
        sbeta: float, optional
            If set, sbeta is fixed to this value, otherwise it
            is calculated from axial induction
        induction: foxes.core.AxialInductionModel or str
            The axial induction model
        wake_k: dict, optional
            Parameters for the WakeK class

        """
        super().__init__(wind_superposition=superposition)
        self.sbeta_factor = sbeta_factor
        self.sbeta = sbeta
        self.induction = induction
        self.wake_k = WakeK(**wake_k)

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.wind_superposition}, induction={iname}, "
        s += self.wake_k.repr() + ")"
        return s

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
        smdls = super().sub_models() + [self.wake_k]
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

    def calc_amplitude_sigma(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """
        Calculate the amplitude and the sigma,
        both depend only on x (not on r).

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
            The index in the downwind order
        x: numpy.ndarray
            The x values, shape: (n_states, n_targets)

        Returns
        -------
        amsi: tuple
            The amplitude and sigma, both numpy.ndarray
            with shape (n_st_sel,)
        st_sel: numpy.ndarray of bool
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)

        """
        assert not isinstance(self.induction, str)

        # get ct:
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        # select targets:
        st_sel = (x > 1e-8) & (ct > 1e-8)
        if np.any(st_sel):
            # apply selection:
            x = x[st_sel]
            ct = ct[st_sel]

            # get D:
            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # get k:
            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )

            # calculate sigma:
            if self.sbeta is None:
                # beta = 0.5 * (1 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)
                a = self.induction.ct2a(ct)
                beta = np.maximum((1 - a) / (1 - 2 * a), 0)
                fsbeta = self.sbeta_factor * np.sqrt(beta)
                del beta, a
            else:
                fsbeta = self.sbeta
            sigma = k * x + fsbeta * D
            del fsbeta

            # calculate amplitude:
            ct_eff = ct / (8 * (sigma / D) ** 2)
            ampld = np.maximum(-2 * self.induction.ct2a(ct_eff), -0.9999)

        # case no targets:
        else:
            st_sel = np.zeros_like(x, dtype=bool)
            n_sp = np.sum(st_sel)
            ampld = np.zeros(n_sp, dtype=config.dtype_double)
            sigma = np.zeros(n_sp, dtype=config.dtype_double)

        return {FV.WS: (ampld, sigma)}, st_sel
