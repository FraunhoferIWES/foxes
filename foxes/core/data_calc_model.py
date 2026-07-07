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
    load_mode: str
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
        args: tuple, optional
            Additional parameters for constructor
        load_mode: str
            The data loading mode, e.g. 'preload'
        kwargs: dict, optional
            Additional parameters for constructor

        """
        super().__init__(*args, **kwargs)
        self.load_mode = load_mode

    @abstractmethod
    def output_coords(self) -> tuple[str, ...]:
        """
        Gets the coordinates of all output arrays

        Returns
        -------
        dims: tuple of str
            The coordinates of all output arrays

        """
        pass

    def load_chunk_data(self, algo: Algorithm, *data: Data) -> None:
        """
        Load chunk data according to load mode.

        This function adds data to mdata.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of foxes.core.Data, optional
            The input data, typically either (mdata, fdata) in
            the case of farm calculations, or (mdata, fdata, tdata)
            for point data calculations

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
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of foxes.core.Data, optional
            The input data, typically either (mdata, fdata) in
            the case of farm calculations, or (mdata, fdata, tdata)
            for point data calculations
        parameters: dict, optional
            The calculation parameters

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray

        """
        self.load_chunk_data(algo, *data)
        return {}
