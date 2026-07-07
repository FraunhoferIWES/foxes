from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from foxes.utils import new_instance

from .farm_data_model import FarmDataModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class TurbineModel(FarmDataModel):
    """
    Abstract base class for turbine models.

    Turbine models are FarmDataModels that run
    on a selection of turbines.

    :group: core

    """

    @abstractmethod
    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        pass

    @classmethod
    def new(
        cls,
        tmodel_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> "TurbineModel":
        """
        Run-time turbine model factory.

        Parameters
        ----------
        tmodel_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, tmodel_type, *args, **kwargs)
