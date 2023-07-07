from .turbine_model import TurbineModel
import foxes.constants as FC


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

    :group: core

    """

    def __init__(self, name=None, D=None, H=None, P_nominal=None, P_unit="kW"):
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

        """
        super().__init__()

        self.name = name if name is not None else type(self).__name__
        self.D = D
        self.H = H
        self.P_nominal = P_nominal
        self.P_unit = P_unit

        if P_unit not in FC.P_UNITS:
            raise KeyError(
                f"Turbine type '{self.name}': Unkown P_unit '{P_unit}', expecting {list(FC.P_UNITS.keys())}"
            )
