from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast

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
        Run the main model calculation.

        This function is executed on a single chunk of data. All computations
        should be based on NumPy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        st_sel
            The state-turbine selection mask with shape ``(n_states, n_turbines)``.

        Returns
        -------
        results
            The resulting data keyed by output variable name. Values are NumPy
            arrays with shape ``(n_states, n_turbines)``.

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
        Create a turbine model instance at runtime.

        Parameters
        ----------
        tmodel_type
            The selected derived class name.
        args
            Additional positional arguments for the constructor.
        kwargs
            Additional keyword arguments for the constructor.

        """
        return cast(TurbineModel, new_instance(cls, tmodel_type, *args, **kwargs))
