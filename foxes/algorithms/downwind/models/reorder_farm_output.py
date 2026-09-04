from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING

from foxes.core import FarmDataModel
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class ReorderFarmOutput(FarmDataModel):
    """
    Reorders final results to state-turbine dimensions
    """

    def __init__(self, outputs: list[str] | None) -> None:
        """
        Parameters
        ----------
        outputs
            The output variables, or None for defaults
        """
        super().__init__()
        self.outputs = outputs

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
        return self.outputs if self.outputs is not None else algo.farm_vars

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
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

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values with shape (n_states, n_turbines)

        """
        ssel = fdata[FV.ORDER_SSEL]
        order_inv = fdata[FV.ORDER_INV]

        out = {}
        for v in self.output_farm_vars(algo):
            if (
                v != FV.ORDER
                and fdata[v].shape[1] > 1
                and np.any(fdata[v] != fdata[v][0, 0, None, None])
            ):
                out[v] = fdata[v][ssel, order_inv]
            else:
                out[v] = fdata[v]
        return out
