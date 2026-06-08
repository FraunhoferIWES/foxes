from abc import abstractmethod
import numpy as np

from foxes.utils import new_instance
import foxes.constants as FC

from .turbine_model import TurbineModel


class TurbineType(TurbineModel):
    """
    Abstract base class for turbine type models.

    Rotor diameter and hub height can be overwritten
    by individual settings in the Turbine object.

    Attributes
    ----------
    name: str
        The model name
    D: float
        The rotor diameter
    H: float
        The hub height
    P_nominal: float
        The nominal power in kW
    P_unit: str
        The unit of power
    rho_corr_P: str
        The type of air density correction for the power curve, choices:
        "rho": apply the air density ratio (rho / rho_ref) as a factor to the power curve
        "rews": correct the rotor equivalent wind speed by the air density ratio (rho / rho_ref)^3
        None: no air density correction applied to the power curve
    rho_corr_ct: str
        The type of air density correction for the ct curve, choices:
        "rho": apply the air density ratio (rho / rho_ref) as a factor to the ct curve
        "rews": correct the rotor equivalent wind speed by the air density ratio (rho / rho_ref)^2
        None: no air density correction applied to the ct curve
    yawm_corr_P: str, optional
        The type of yaw misalignment correction for the power curve, choices:
        "factor": apply a correction factor to the power curve
        "wind_speed": correct the rotor equivalent wind speed by a yaw misalignment factor
        None: no yaw misalignment correction applied to the power curve
    yawm_corr_ct: str, optional
        The type of yaw misalignment correction for the ct curve, choices:
        "factor": apply a correction factor to the ct curve
        "wind_speed": correct the rotor equivalent wind speed by a yaw misalignment factor
        None: no yaw misalignment correction applied to the ct curve
    yawm_corr_p_P: float, optional
        The exponent for yaw dependency of P
    yawm_corr_p_ct: float, optional
        The exponent for yaw dependency of ct

    :group: core

    """

    def __init__(
        self,
        name=None,
        D=None,
        H=None,
        P_nominal=None,
        P_unit="kW",
        rho_corr_P="wind_speed",
        rho_corr_ct=None,
        yawm_corr_P="wind_speed",
        yawm_corr_ct="wind_speed",
        yawm_corr_p_P=1.88,
        yawm_corr_p_ct=1.0,
    ):
        """
        Constructor.

        Parameters
        ----------
        name: str, optional
            The model name
        D: float, optional
            The rotor diameter
        H: float, optional
            The hub height
        P_nominal: float, optional
            The nominal power in kW
        P_unit: str
            The unit of power, choices:
            W, kW, MW, GW
        rho_corr_P: str, optional
            The type of air density correction for the power curve, choices:
            "factor": apply the air density ratio (rho / rho_ref) as a factor to the power curve
            "wind_speed": correct the rotor equivalent wind speed by the air density ratio (rho / rho_ref)^3
            None: no air density correction applied to the power curve
        rho_corr_ct: str, optional
            The type of air density correction for the ct curve, choices:
            "factor": apply the air density ratio (rho / rho_ref) as a factor to the ct curve
            "wind_speed": correct the rotor equivalent wind speed by the air density ratio (rho / rho_ref)^2
            None: no air density correction applied to the ct curve
        yawm_corr_P: str, optional
            The type of yaw misalignment correction for the power curve, choices:
            "factor": apply a correction factor to the power curve
            "wind_speed": correct the rotor equivalent wind speed by a yaw misalignment factor
            None: no yaw misalignment correction applied to the power curve
        yawm_corr_ct: str, optional
            The type of yaw misalignment correction for the ct curve, choices:
            "factor": apply a correction factor to the ct curve
            "wind_speed": correct the rotor equivalent wind speed by a yaw misalignment factor
            None: no yaw misalignment correction applied to the ct curve
        yawm_corr_p_P: float, optional
            The exponent for yaw dependency of P
        yawm_corr_p_ct: float, optional
            The exponent for yaw dependency of ct
        p_P: float
            The exponent for yaw dependency of P

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

    def __repr__(self):
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}"
        return f"{type(self).__name__}({a})"

    @abstractmethod
    def needs_rews2(self):
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag: bool
            True if REWS2 is required

        """
        pass

    @abstractmethod
    def needs_rews3(self):
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag: bool
            True if REWS3 is required

        """
        pass

    def modify_cutin(self, modify_ct, modify_P):
        """
        Modify the data such that a discontinuity
        at cutin wind speed is avoided

        Parameters
        ----------
        variable: str
            The target variable
        modify_ct: bool
            Flag for modification of the ct curve
        modify_P: bool
            Flag for modification of the power curve

        """
        if modify_ct or modify_P:
            raise NotImplementedError(
                f"Turbine type '{self.name}': Continuous cutin not implemented for modify_ct = {modify_ct}, modify_P = {modify_P}"
            )

    def get_rho_yawm_corrections(
        self,
        rews_P,
        rews_ct,
        rho,
        rho_ref=None,
        yawm=None,
    ):
        """
        Compute air density and yaw corrections.

        Parameters
        ----------
        rews_P: np.ndarray
            The equivalent wind speeds for the power curve
        rews_ct: np.ndarray
            The equivalent wind speeds for the ct curve
        rho: np.ndarray
            The air density values in kg/m^3
        rho_ref: np.ndarray or float, optional
            The reference air density in kg/m^3 for the correction
        yawm: np.ndarray, optional
            The yaw misalignment values in degrees

        Returns
        -------
        rews_P_corr: np.ndarray
            The corrected equivalent wind speeds for the power curve
        rews_ct_corr: np.ndarray
            The corrected equivalent wind speeds for the ct curve
        factor_P: np.ndarray
            The correction factor for the power curve
        factor_ct: np.ndarray
            The correction factor for the ct curve

        """
        factor_P = 1.0
        factor_ct = 1.0

        # compute air density correction for power curve:
        if rho_ref is None or self.rho_corr_P is None:
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
        if rho_ref is None or self.rho_corr_ct is None:
            pass
        elif self.rho_corr_ct == "factor":
            factor_ct = rho / rho_ref
        elif self.rho_corr_ct == "wind_speed":
            rews_ct *= (rho / rho_ref) ** 0.5
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown rho_corr_ct '{self.rho_corr_ct}', expecting 'factor', 'wind_speed' or None"
            )

        # compute yaw misalignment correction for power curve:
        if yawm is None or self.yawm_corr_P is None:
            pass
        elif self.yawm_corr_P == "factor":
            cosm = np.cos(yawm / 180 * np.pi)
            factor_P *= cosm**self.yawm_corr_p_P
        elif self.yawm_corr_P == "wind_speed":
            cosm = np.cos(yawm / 180 * np.pi)
            rews_P *= (cosm**self.yawm_corr_p_P) ** (1.0 / 3.0)
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown yawm_corr_P '{self.yawm_corr_P}', expecting 'factor', 'wind_speed' or None"
            )

        # compute yaw misalignment correction for ct curve:
        if yawm is None or self.yawm_corr_ct is None:
            pass
        elif self.yawm_corr_ct == "factor":
            cosm = np.cos(yawm / 180 * np.pi)
            factor_ct *= cosm**self.yawm_corr_p_ct
        elif self.yawm_corr_ct == "wind_speed":
            cosm = np.cos(yawm / 180 * np.pi)
            rews_ct *= (cosm**self.yawm_corr_p_ct) ** 0.5
        else:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown yawm_corr_ct '{self.yawm_corr_ct}', expecting 'factor', 'wind_speed' or None"
            )

        return rews_P, rews_ct, factor_P, factor_ct

    @classmethod
    def new(cls, ttype_type, *args, **kwargs):
        """
        Run-time turbine type factory.

        Parameters
        ----------
        ttype_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, ttype_type, *args, **kwargs)
