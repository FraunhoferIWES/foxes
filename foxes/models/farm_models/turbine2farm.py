from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import FarmModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import Model
    from foxes.core.turbine_model import TurbineModel


class Turbine2FarmModel(FarmModel):
    """
    Wrapper that promotes turbine models
    into farm models, simply by selecting
    all turbines.

    Attributes
    ----------
    turbine_model
        The turbine model


    """

    def __init__(self, turbine_model: TurbineModel) -> None:
        """
        Constructor.

        Parameters
        ----------
        turbine_model
            The turbine model

        """
        super().__init__()
        self.turbine_model = turbine_model
        self.name = turbine_model.name + "_t2f"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.turbine_model})"

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return [self.turbine_model]

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        return self.turbine_model.output_farm_vars(algo)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        **parameters: Any,
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        **parameters
            Init parameters forwarded to the turbine model

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values have shape (n_states, n_turbines)

        """
        n_states = algo.n_states
        n_turbines = algo.n_turbines
        assert n_states is not None and n_turbines is not None
        s = np.ones((n_states, n_turbines), dtype=np.bool_)
        return self.turbine_model.calculate(algo, mdata, fdata, st_sel=s, **parameters)
