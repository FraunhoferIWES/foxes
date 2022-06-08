from foxes.core.turbine_model import TurbineModel

class TurbineType(TurbineModel):
    """
    Abstract base class for turbine type models.

    Rotor diameter and hub height can be overwritten
    by individual settings in the Turbine object.

    Parameters
    ----------
    name : str
        The model name
    D : float
        The rotor diameter
    H : float, optional
        The hub height
    
    Attributes
    ----------
    name : str
        The model name
    D : float
        The rotor diameter
    H : float, optional
        The hub height    

    """

    def __init__(
        self,
        name,
        D,
        H=None
    ):
        super().__init__()
        self.name = name
        self.D    = D
        self.H    = H
