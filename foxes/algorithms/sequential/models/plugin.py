from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarray import Dataset
    from foxes.algorithms.sequential.sequential import Sequential


class SequentialPlugin:
    """
    Base class for plugins that are
    updated with each sequential iteration

    Parameters
    ----------
    algo: foxes.algorithms.sequential.Sequential
        The sequential algorithm

    :group: algorithms.sequential.models

    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        self.algo: Sequential | None = None

    def initialize(self, algo: Sequential) -> None:
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The current sequential algorithm

        """
        self.algo = algo

    def update(
        self, algo: Sequential, fres: Dataset, pres: Dataset | None = None
    ) -> None:
        """
        Updates data based on current iteration

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The latest sequential algorithm
        fres: xarray.Dataset
            The latest farm results
        pres: xarray.Dataset, optional
            The latest point results

        """
        self.algo = algo

    def finalize(self, algo: Sequential) -> None:
        """
        Finalize data based on the final iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The final sequential algorithm

        """
        self.algo = algo
