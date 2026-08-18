from __future__ import annotations

from typing import Any

from .segregated import PartialSegregated
from foxes.models.rotor_models.grid import GridRotor


class PartialGrid(PartialSegregated):
    """
    Partial wakes on a grid rotor that may
    differ from the one in the algorithm.


    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        args
            Parameters for GridRotor
        kwargs
            Parameters for GridRotor

        """
        super().__init__(GridRotor(*args, calc_vars=[], **kwargs))
