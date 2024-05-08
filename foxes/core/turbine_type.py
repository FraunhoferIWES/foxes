import foxes.constants as FC
from foxes.utils import all_subclasses

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

    def __repr__(self):
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}"
        return f"{type(self).__name__}({a})"

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

        if ttype_type is None:
            return None

        allc = all_subclasses(cls)
        found = ttype_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == ttype_type:
                    return scls(*args, **kwargs)

        else:
            estr = "Turbine type class '{}' is not defined, available types are \n {}".format(
                ttype_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
