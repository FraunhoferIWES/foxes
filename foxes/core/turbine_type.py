from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
import numpy as np
from typing import Any, cast

from foxes.utils import new_instance
import foxes.constants as FC

from .turbine_model import TurbineModel


class TurbineType(TurbineModel):
    """
    Abstract base class for turbine type models.

    Rotor diameter and hub height can be overwritten
    by individual settings in the Turbine object.
    """

    def __init__(
        self,
        name: str | None = None,
        D: float | np.ndarray | None = None,
        H: float | np.ndarray | None = None,
        P_nominal: float | None = None,
        P_unit: str = "kW",
        rho_corr_P: str | None = "wind_speed",
        rho_corr_ct: str | None = None,
        yawm_corr_P: str | None = "wind_speed",
        yawm_corr_ct: str | None = "wind_speed",
        yawm_corr_p_P: float = 1.88,
        yawm_corr_p_ct: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        name
            The model name.
        D
            The rotor diameter.
        H
            The hub height.
        P_nominal
            The nominal power in kW.
        P_unit
            The power unit, one of ``W``, ``kW``, ``MW``, or ``GW``.
        rho_corr_P
            The air-density correction mode for the power curve.
        rho_corr_ct
            The air-density correction mode for the thrust curve.
        yawm_corr_P
            The yaw-misalignment correction mode for the power curve.
        yawm_corr_ct
            The yaw-misalignment correction mode for the thrust curve.
        yawm_corr_p_P
            The exponent for yaw dependency of power.
        yawm_corr_p_ct
            The exponent for yaw dependency of thrust.
        """
        super().__init__()

        self.name = name if name is not None else type(self).__name__
        self.D = D
        self.H = H
        self.P_nominal = P_nominal
        self.P_unit = P_unit
        self.rho_corr_P = rho_corr_P
        self.rho_corr_ct = rho_corr_ct
        self.yawm_corr_P = yawm_corr_P
        self.yawm_corr_ct = yawm_corr_ct
        self.yawm_corr_p_P = yawm_corr_p_P
        self.yawm_corr_p_ct = yawm_corr_p_ct
        if P_unit not in FC.P_UNITS:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown P_unit '{P_unit}', expecting {list(FC.P_UNITS.keys())}"
            )

    def __repr__(self) -> str:
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}"
        return f"{type(self).__name__}({a})"

    @abstractmethod
    def needs_rews2(self) -> bool:
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        needs_rews2
            True if REWS2 is required

        """
        pass

    @abstractmethod
    def needs_rews3(self) -> bool:
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        needs_rews3
            True if REWS3 is required

        """
        pass

    def modify_cutin(self, modify_ct: bool, modify_P: bool) -> None:
        """
        Modify the data to avoid a discontinuity at cut-in wind speed.

        Parameters
        ----------
        variable
            The target variable.
        modify_ct
            Flag for modifying the thrust curve.
        modify_P
            Flag for modifying the power curve.

        """
        if modify_ct or modify_P:
            raise NotImplementedError(
                f"Turbine type '{self.name}': Continuous cutin not implemented for modify_ct = {modify_ct}, modify_P = {modify_P}"
            )

    def get_rho_yawm_corrections(
        self,
        rews_P: np.ndarray,
        rews_ct: np.ndarray,
        rho: np.ndarray | None = None,
        rho_ref: np.ndarray | float | None = None,
        yawm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float | np.ndarray, float | np.ndarray]:
        """
        Compute air density and yaw corrections.

        Parameters
        ----------
        rews_P
            The equivalent wind speeds for the power curve
        rews_ct
            The equivalent wind speeds for the ct curve
        rho
            The air density values in kg/m^3
        rho_ref
            The reference air density in kg/m^3 for the correction
        yawm
            The yaw misalignment values in degrees

        Returns
        -------
        rews_P_corr
            The corrected equivalent wind speeds for the power curve
        rews_ct_corr
            The corrected equivalent wind speeds for the ct curve
        factor_P
            The correction factor for the power curve
        factor_ct
            The correction factor for the ct curve

        """
        factor_P: float | np.ndarray = 1.0
        factor_ct: float | np.ndarray = 1.0

        # compute air density correction for power curve:
        if rho is None or rho_ref is None or self.rho_corr_P is None:
            pass
        elif self.rho_corr_P == "factor":
            factor_P = rho / rho_ref
        elif self.rho_corr_P == "wind_speed":
            rews_P *= (rho / rho_ref) ** (1.0 / 3.0)
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown rho_corr_P '{self.rho_corr_P}', expecting 'factor', 'wind_speed' or None"
            )

        # compute air density correction for ct curve:
        if rho is None or rho_ref is None or self.rho_corr_ct is None:
            pass
        else:
            raise NotImplementedError(
                f"Turbine type '{self.name}': Air density correction for ct curve not implemented"
            )
        # elif self.rho_corr_ct == "factor":
        #    factor_ct = rho / rho_ref
        # elif self.rho_corr_ct == "wind_speed":
        #    rews_ct *= (rho / rho_ref) ** 0.5
        # else:
        #    raise KeyError(
        #        f"Turbine type '{self.name}': Unkown rho_corr_ct '{self.rho_corr_ct}', expecting 'factor', 'wind_speed' or None"
        #    )

        # compute yaw misalignment correction for power curve:
        if yawm is None or self.yawm_corr_P is None:
            pass
        elif self.yawm_corr_P == "factor":
            # Cosine-based yaw correction is only physically meaningful for positive inflow projection.
            cosm = np.clip(np.cos(yawm / 180 * np.pi), 0.0, None)
            factor_P *= cosm**self.yawm_corr_p_P
        elif self.yawm_corr_P == "wind_speed":
            cosm = np.clip(np.cos(yawm / 180 * np.pi), 0.0, None)
            rews_P *= (cosm**self.yawm_corr_p_P) ** (1.0 / 3.0)
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown yawm_corr_P '{self.yawm_corr_P}', expecting 'factor', 'wind_speed' or None"
            )

        # compute yaw misalignment correction for ct curve:
        if yawm is None or self.yawm_corr_ct is None:
            pass
        elif self.yawm_corr_ct == "factor":
            cosm = np.clip(np.cos(yawm / 180 * np.pi), 0.0, None)
            factor_ct *= cosm**self.yawm_corr_p_ct
        elif self.yawm_corr_ct == "wind_speed":
            cosm = np.clip(np.cos(yawm / 180 * np.pi), 0.0, None)
            rews_ct *= (cosm**self.yawm_corr_p_ct) ** 0.5
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown yawm_corr_ct '{self.yawm_corr_ct}', expecting 'factor', 'wind_speed' or None"
            )

        return rews_P, rews_ct, factor_P, factor_ct

    @classmethod
    def new(
        cls,
        ttype_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> TurbineType:
        """
        Run-time turbine type factory.

        Parameters
        ----------
        ttype_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(TurbineType, new_instance(cls, ttype_type, *args, **kwargs))
