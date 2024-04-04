from foxes.models.partial_wakes.distsliced import PartialDistSlicedWake
from foxes.models.rotor_models.grid import GridRotor


class PartialGrid(PartialDistSlicedWake):
    """
    Partial wakes on a grid rotor that may
    differ from the one in the algorithm.

    :group: models.partial_wakes

    """

    def __init__(
        self, n, rotor_model=None, **kwargs
    ):
        """
        Constructor.

        Parameters
        ----------
        n: int, optional
            The `GridRotor`'s `n` parameter
        rotor_model: foxes.core.RotorModel, optional
            The rotor model, default is the one from the algorithm
        kwargs: dict, optional
            Additional parameters for the `GridRotor`

        """
        super().__init__(n, rotor_model, **kwargs)

        if not isinstance(self.grotor, GridRotor):
            raise ValueError(
                f"Wrong grotor type, expecting {GridRotor.__name__}, got {type(self.grotor).__name__}"
            )
