from __future__ import annotations
# mypy: disable-error-code=override

from typing import TYPE_CHECKING, Any

import foxes.variables as FV
from foxes.core import FarmDataModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData


class SetAmbFarmResults(FarmDataModel):
    """
    This model copies farm data results to ambient results.
    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        super().__init__()
        self.vars: set[str] | None = None

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
        if self.vars is None:
            self.vars = set([v for v in algo.farm_vars if v in FV.var2amb])
        return [FV.var2amb[v] for v in self.vars]

    def calculate(self, algo: Algorithm, mdata: MData, fdata: FData) -> dict[str, Any]:
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
        ovars = self.output_farm_vars(algo)
        assert self.vars is not None
        for v in self.vars:
            fdata.add(FV.var2amb[v], fdata[v].copy(), fdata.dims[v])
        return {v: fdata[v] for v in ovars}
