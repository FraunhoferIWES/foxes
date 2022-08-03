from .turbine_model import TurbineModel


class TurbineType(TurbineModel):
    """
    Abstract base class for turbine type models.

    Rotor diameter and hub height can be overwritten
    by individual settings in the Turbine object.

    Parameters
    ----------
    name : str, optional
        The model name
    D : float, optional
        The rotor diameter
    H : float, optional
        The hub height
    P_nominal : float, optional
        The nominal power in kW

    Attributes
    ----------
    name : str
        The model name
    D : float
        The rotor diameter
    H : float
        The hub height
    P_nominal : float
        The nominal power in kW

    """

    def __init__(self, name=None, D=None, H=None, P_nominal=None):
        super().__init__()

        self.name = name if name is not None else type(self).__name__
        self.D = D
        self.H = H
        self.P_nominal = P_nominal
