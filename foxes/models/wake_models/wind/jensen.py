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
    """

    def __init__(
        self, superposition: str, induction: str = "Betz", **wake_k: Any
    ) -> None:
        """
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

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

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


class JensenTurbOParkWake(TopHatWakeModel):
    """
    Jensen wake model with TurbOPark-like wake growth.

    The model keeps the top-hat wake structure of Jensen's
    centreline formulation, but uses the TurbOPark wake-growth
    expression for the effective wake radius.

    :group: models.wake_models.wind
    """

    def __init__(
        self,
        superposition: str,
        sbeta_factor: float = 0.25,
        c1: float = 1.5,
        c2: float = 0.8,
        induction: str = "Madsen",
        **wake_k: Any,
    ) -> None:
        """
        Parameters
        ----------
        superposition
            The wind deficit superposition
        sbeta_factor
            Factor multiplying sbeta
        c1
            Factor from Frandsen turbulence model
        c2
            Factor from Frandsen turbulence model
        induction
            The induction model
        wake_k
            Parameters for the WakeK class
        """
        super().__init__(wind_superposition=superposition, induction=induction)
        self.sbeta_factor = sbeta_factor
        self.c1 = c1
        self.c2 = c2
        self.wake_k = WakeK(**wake_k)

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.wind_superposition}, induction={iname}, "
        s += self.wake_k.repr() + ")"
        return s

    def waked_variables(self) -> list[str]:
        """
        Returns a list of variable names that are affected by this wake model.

        Returns
        -------
        waked_variables
            A list of variable names affected by this wake model.

        """
        return [FV.WS]

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
        Calculate the wake radius using a TurbOPark-like growth law.

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
        assert not isinstance(self.induction, str)

        wake_r = np.zeros_like(x, dtype=np.float64)
        st_sel = (x > 1e-8) & (ct > 1e-8)

        if np.any(st_sel):
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

            ati = self.get_data(
                FV.AMB_TI,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
            )
            ati = ati[st_sel]

            k = self.wake_k(
                FC.STATE_TARGET,
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                amb_ti=ati,
                upcast=False,
                selection=st_sel,
            )

            alpha = self.c1 * ati
            beta = self.c2 * ati / np.sqrt(ct[st_sel])

            sigma = D * (
                k
                / beta
                * (
                    np.sqrt((alpha + beta * x[st_sel] / D) ** 2 + 1)
                    - np.sqrt(1 + alpha**2)
                    - np.log(
                        (np.sqrt((alpha + beta * x[st_sel] / D) ** 2 + 1) + 1)
                        * alpha
                        / ((np.sqrt(1 + alpha**2) + 1) * (alpha + beta * x[st_sel] / D))
                    )
                )
            )

            wake_r[st_sel] = D / 2 + sigma

        return wake_r

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
            varlue: numpy.ndarray, shape: (n_st_sel,)

        """
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

        assert not isinstance(self.induction, str)
        twoa = 2 * self.induction.ct2a(ct)

        return {FV.WS: -((R / wake_r) ** 2) * twoa}
