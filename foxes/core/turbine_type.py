from foxes.core.turbine_model import TurbineModel

class TurbineType(TurbineModel):

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
