from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from foxes.utils import new_instance
# from foxes.algorithms import Downwind
# from foxes.input.states import ScanStates

if TYPE_CHECKING:
    pass

from .single_turbine_wake_model import SingleTurbineWakeModel


class TurbineInductionModel(SingleTurbineWakeModel):
    """
    Abstract base class for turbine induction models.


    """

    @property
    def affects_downwind(self) -> bool:
        """
        Flag for downwind or upwind effects
        on other turbines

        Returns
        -------
        affects_downwind
            Flag for downwind effects by this model

        """
        return False

    @classmethod
    def new(
        cls,
        wmodel_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> TurbineInductionModel:
        """
        Run-time turbine induction model factory.

        Parameters
        ----------
        wmodel_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor

        """
        return cast(
            TurbineInductionModel, new_instance(cls, wmodel_type, *args, **kwargs)
        )
