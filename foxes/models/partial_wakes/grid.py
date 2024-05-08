from .segregated import PartialSegregated
from foxes.models.rotor_models.grid import GridRotor


class PartialGrid(PartialSegregated):
    """
    Partial wakes on a grid rotor that may
    differ from the one in the algorithm.

    :group: models.partial_wakes

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Parameters for GridRotor
        kwargs: dict, optional
            Parameters for GridRotor

        """
        super().__init__(GridRotor(*args, calc_vars=[], **kwargs))
