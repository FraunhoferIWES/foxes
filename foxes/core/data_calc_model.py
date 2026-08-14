from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .model import Model

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import Data


class DataCalcModel(Model):
    """
    Abstract base class for models
    that run calculations based on model data.

    Attributes
    ----------
    load_mode
        The data loading mode

    :group: core

    """

    def __init__(
        self,
        *args: Any,
        load_mode: str = "preload",
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        args
            Additional positional arguments for the constructor.
        load_mode
            The data loading mode, e.g. ``"preload"``.
        kwargs
            Additional keyword arguments for the constructor.

        """
        super().__init__(*args, **kwargs)
        self.load_mode = load_mode

    @abstractmethod
    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims
            The coordinates of all output arrays

        """
        pass

    def load_chunk_data(self, algo: Algorithm, *data: Data) -> None:
        """
        Load chunk data according to the configured load mode.

        This function adds data to the model data container.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data
            Input data, typically either ``(mdata, fdata)`` for farm
            calculations or ``(mdata, fdata, tdata)`` for point data
            calculations.

        """
        for m in self.sub_models():
            load_chunk_data = getattr(m, "load_chunk_data", None)
            if callable(load_chunk_data):
                load_chunk_data(algo, *data)

    @abstractmethod
    def calculate(
        self,
        algo: Algorithm,
        *data: Data,
        **parameters: Any,
    ) -> dict[str, np.ndarray]:
        """
        Execute the main model calculation.

        This function is executed on a single chunk of data. All computations
        should be based on NumPy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data
            Input data, typically either ``(mdata, fdata)`` for farm
            calculations or ``(mdata, fdata, tdata)`` for point data
            calculations.
        parameters
            Calculation parameters.

        Returns
        -------
        results
            The resulting data. Keys are output variable names and values are
            NumPy arrays.

        """
        self.load_chunk_data(algo, *data)
        return {}
