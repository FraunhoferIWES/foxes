from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import WakeK
from foxes.models.wake_models.top_hat import TopHatWakeModel
import foxes.variables as FV
import foxes.constants as FC

from foxes.core.algorithm import Algorithm
from foxes.core.data import FData, MData, TData


class JensenWake(TopHatWakeModel):
    """
    The Jensen wake model.

    Attributes
    ----------
    wake_k
        Handler for the wake growth parameter k


    """

    def __init__(
        self, superposition: str, induction: str = "Betz", **wake_k: Any
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        superposition
            The wind deficit superposition
        induction
            The induction model
        wake_k
            Parameters for the WakeK class

        """
        super().__init__(wind_superposition=superposition, induction=induction)
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
        dws
            If True, this model affects wind speed

        """
        return True

    def calc_wake_radius(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
        ct: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the wake radius, depending on x only (not r).

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
            The index in the downwind order
        x
            The x values, shape: (n_states, n_targets)
        ct
            The ct values of the wake-causing turbines,
            shape: (n_states, n_targets)

        Returns
        -------
        wake_r
            The wake radii, shape: (n_states, n_targets)

        """
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        k = self.wake_k(
            FC.STATE_TARGET,
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        return D / 2 + k * x

    def calc_centreline(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        st_sel: np.ndarray,
        x: np.ndarray,
        wake_r: np.ndarray,
        ct: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Calculate centre line results of wake deltas.

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
            The index in the downwind order
        st_sel
            The state-target selection, for which the wake
            is non-zero, shape: (n_states, n_targets)
        x
            The x values, shape: (n_st_sel,)
        wake_r
            The wake radii, shape: (n_st_sel,)
        ct
            The ct values of the wake-causing turbines,
            shape: (n_st_sel,)

        Returns
        -------
        cl_del
            The centre line wake deltas. Key: variable name str,
            varlue

        """
        assert not isinstance(self.induction, str)

        R = (
            self.get_data(
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
            / 2
        )

        twoa = 2 * self.induction.ct2a(ct)

        return {FV.WS: -((R / wake_r) ** 2) * twoa}
