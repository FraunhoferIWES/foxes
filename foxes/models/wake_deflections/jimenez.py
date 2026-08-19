from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from foxes.core.wake_deflection import WakeDeflection
from foxes.algorithms import Sequential
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class JimenezDeflection(WakeDeflection):
    """
    Yawed rotor wake defection according to the Jimenez model

    Notes
    -----
    Reference:
    Jiménez, Á., Crespo, A. and Migoya, E. (2010), Application of a LES technique to characterize
    the wake deflection of a wind turbine in yaw. Wind Energ., 13: 559-572. doi:10.1002/we.380


    Attributes
    ----------
    rotate
        If True, rotate local wind vector at evaluation points.
        If False, multiply wind speed with cos(angle) instead.
        If None, do not modify the wind vector, only the path.
    beta
        The beta coefficient of the Jimenez model
    step_x
        The x step in m for integration


    """

    def __init__(
        self,
        rotate: bool | None = True,
        beta: float = 0.1,
        step_x: float = 10.0,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        rotate
            If True, rotate local wind vector at evaluation points.
            If False, multiply wind speed with cos(angle) instead.
            If None, do not modify the wind vector, only the path.
        beta
            The beta coefficient of the Jimenez model
        step_x
            The x step in m for integration

        """
        super().__init__()
        self.rotate = rotate
        self.beta = beta
        self.step_x = step_x

    def __repr__(self) -> str:
        s = f"{type(self).__name__}("
        s += f"rotate={self.rotate}, beta={self.beta}, step_x={self.step_x}"
        s += ")"
        return s

    @property
    def has_uv(self) -> bool:
        """
        This model uses wind vector data

        Returns
        -------
        has_uv
            Flag for wind vector data

        """
        return self.rotate is not None and self.rotate

    def calc_deflection(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        coos: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the wake deflection.

        This function optionally adds FC.WDEFL_ROT_ANGLE or
        FC.WDEFL_DWS_FACTOR to the tdata.

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
        coos
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        Returns
        -------
        coos
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """

        if FV.YAWM not in fdata:
            return coos

        # take rotor average:
        xyz = np.einsum("stpd,p->std", coos, tdata[FC.TWEIGHTS])
        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

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

        # get gamma:
        gamma = self.get_data(
            FV.YAWM,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        sel = (x > 1e-8) & (x < 1e10) & (ct > 1e-8) & (np.abs(gamma) > 1e-8)
        delwd = np.zeros_like(coos[..., 0])
        n_sel = np.sum(sel)
        if n_sel > 0:
            # apply selection:
            gamma = np.deg2rad(gamma[sel])
            ct = ct[sel]
            x = x[sel]

            # get rotor diameter:
            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                algo=algo,
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                upcast=True,
                selection=sel,
            )[:, None]

            # define x path:
            xmax = np.max(x)
            n_x = int(xmax / self.step_x)
            if xmax > n_x * self.step_x:
                n_x += 1
            delx = np.arange(n_x + 1) * self.step_x
            delx = np.minimum(delx[None, :], x[:, None])
            dx = delx[:, 1:] - delx[:, :-1]
            delx = delx[:, :-1]

            # integrate deflection of y along the x path:
            alpha0 = (
                -(np.cos(gamma[:, None]) ** 2)
                * np.sin(gamma[:, None])
                * ct[:, None]
                / 2
            )
            y[sel] += np.sum(
                np.tan(alpha0 / (1 + self.beta * delx / D) ** 2) * dx, axis=-1
            )
            del delx, dx
            coos[..., 1] = y[:, :, None]

            # calculate wind vector modification at evaluation points:
            if self.rotate is not None:
                # delta wd at evaluation points, if within wake radius:
                r2 = (y[sel, None] ** 2 + z[sel, None] ** 2) / D**2
                WD2 = (1 + self.beta * x[:, None] / D) ** 2
                delwd[sel] = np.where(r2 <= WD2 / 4, alpha0 / WD2, 0)

                if self.rotate:
                    tdata[FC.WDEFL_ROT_ANGLE] = np.rad2deg(delwd)
                else:
                    tdata[FC.WDEFL_DWS_FACTOR] = np.cos(delwd)

        return coos

    def get_yaw_alpha_seq(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Computes sequential wind vector rotation angles.

        Wind vector rotation angles are computed at the
        current trace points due to a yawed rotor
        for sequential runs.

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
        x
            The distance from the wake causing rotor
            for the first n_times subsequent time steps,
            shape: (n_times,)

        Returns
        -------
        alpha
            The delta WD result at the x locations,
            shape: (n_times,)

        """
        assert isinstance(algo, Sequential), (
            f"Wake deflection '{self.name}' requires Sequential algorithm, got '{type(algo).__name__}'"
        )

        n_times = len(x)

        def _get_data(var: str) -> np.ndarray:
            data = algo.farm_results_downwind[var].to_numpy()[:n_times, downwind_index]
            data[-1] = fdata[var][0, downwind_index]
            return data

        gamma = _get_data(FV.YAWM)
        ct = _get_data(FV.CT)
        alpha = np.zeros_like(gamma)

        sel = (ct > 1e-8) & (np.abs(gamma) > 1e-8)
        if np.any(sel):
            D = _get_data(FV.D)[sel]
            gamma = np.deg2rad(gamma[sel])
            ct = ct[sel]
            x = x[sel]

            alpha[sel] = np.rad2deg(
                -(np.cos(gamma) ** 2)
                * np.sin(gamma)
                * ct
                / 2
                / (1 + self.beta * x / D) ** 2
            )

        return alpha
