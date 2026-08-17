from __future__ import annotations
# mypy: disable-error-code=override

from foxes.core import PointDataModel
import foxes.variables as FV
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class WakeDeltas(PointDataModel):
    """
    This point model simply subtracts ambient results
    from waked results.

    Attributes
    ----------
    vars
        The variables
    normalize
        Divide resulting deltas by ambient values

    :group: models.point_models

    """

    def __init__(self, vars: list[str], normalize: bool = False) -> None:
        """
        Constructor.

        Parameters
        ----------
        vars
            The variables
        normalize
            Divide resulting deltas by ambient values

        """
        super().__init__()
        self.vars = vars
        self.normalize = normalize

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        return [f"DELTA_{v}" for v in self.vars]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        pdata: TData,
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
        tdata
            The target point data

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """

        out = {f"DELTA_{v}": pdata[v] - pdata[FV.var2amb[v]] for v in self.vars}

        if self.normalize:
            for v in self.vars:
                out[v] /= pdata[FV.var2amb[v]]

        return out
