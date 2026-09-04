from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import WakeK
from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import Model


class CrespoHernandezTIWake(TopHatWakeModel):
    """
    The Crespo and Hernandez TI empirical correlation

    Notes
    -----
    Reference:
    "Turbulence characteristics in wind-turbine wakes"
    A. Crespo, J. Hernandez
    https://doi.org/10.1016/0167-6105(95)00033-X

    For the wake diameter we use Eqns. (17), (15), (4), (5) from
            doi:10.1088/1742-6596/625/1/012039
    """

    def __init__(
        self,
        superposition: str,
        use_ambti: bool = False,
        sbeta_factor: float = 0.25,
        near_wake_D: float | None = None,
        a_near: float = 0.362,
        a_far: float = 0.73,
        e1: float = 0.83,
        e2: float = -0.0325,
        e3: float = -0.32,
        induction: str = "Betz",
        **wake_k: Any,
    ) -> None:
        """
        Parameters
        ----------
        superposition
            The TI wake superposition.
        k
            The wake growth parameter k. If not given here
            it will be searched in the farm data.
        use_ambti
            Flag for using ambient TI instead of local
            wake corrected TI
        sbeta_factor
            Factor multiplying sbeta
        near_wake_D
            The near wake distance in units of D,
            calculated from TI and ct if not given here
        a_near
            Model parameter
        a_far
            Model parameter
        e1
            Model parameter
        e2
            Model parameter
        e3
            Model parameter
        k_var
            The variable name for k
        induction
            The induction model
        wake_k
            Parameters for the WakeK class
        """
        super().__init__(
            other_superpositions={FV.TI: superposition}, induction=induction
        )

        self.a_near = a_near
        self.a_far = a_far
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3
        self.use_ambti = use_ambti
        self.sbeta_factor = sbeta_factor
        self.near_wake_D = near_wake_D
        self.wake_k = WakeK(**wake_k)

    def __repr__(self) -> str:
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        s = f"{type(self).__name__}"
        s += f"({self.other_superpositions[FV.TI]}, induction={iname}, "
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
        return [FV.TI]

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return super().sub_models() + [self.wake_k]

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
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return {FV.TI: np.zeros_like(tdata[FC.TARGETS][..., 0])}

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
        assert not isinstance(self.induction, str)

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
        )

        # get k:
        k = self.wake_k(
            FC.STATE_TARGET,
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=False,
        )

        # calculate:
        a = self.induction.ct2a(ct)
        beta = np.maximum((1 - a) / (1 - 2 * a), 0)
        radius = 2 * (k * x + self.sbeta_factor * np.sqrt(beta) * D)

        return radius

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

        # prepare:
        n_targts = np.sum(st_sel)
        TI = FV.AMB_TI if self.use_ambti else FV.TI

        # get D:
        D = np.asarray(
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
        )

        # get TI:
        ti = np.asarray(
            self.get_data(
                TI,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=False,
                selection=st_sel,
            )
        )

        # avoid zero ti values:
        ti = np.maximum(ti, 1e-10)

        # calculate induction factor:
        twoa = np.asarray(2 * self.induction.ct2a(ct))

        # prepare output:
        wake_deltas = np.zeros(n_targts, dtype=config.dtype_double)

        # calc near wake length, if not given
        if self.near_wake_D is None:
            near_wake_D = (
                2**self.e1
                * self.a_near
                / (self.a_far * ti**self.e2)
                * twoa ** (1 - self.e1)
            ) ** (1 / self.e3)
        else:
            near_wake_D = np.full_like(x, self.near_wake_D, dtype=config.dtype_double)

        # calc near wake:
        sel = x < near_wake_D * D
        if np.any(sel):
            wake_deltas[sel] = self.a_near * twoa[sel]

        # calc far wake:
        if np.any(~sel):
            # calculate delta:
            #
            # Note the sign flip of the exponent ti[~sel]**(-0.0325)
            # compared to the original paper. This was found in
            # https://doi.org/10.1016/j.jweia.2018.04.010, Eq. (46)
            # Without this flip the near and far wake areas are not
            # smoothly connected.
            #
            wake_deltas[~sel] = (
                self.a_far
                * (twoa[~sel] / 2) ** self.e1
                * ti[~sel] ** self.e2
                * (x[~sel] / D[~sel]) ** self.e3
            )

        return {FV.TI: wake_deltas}
