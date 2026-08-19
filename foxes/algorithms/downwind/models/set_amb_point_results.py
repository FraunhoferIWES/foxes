from __future__ import annotations
# mypy: disable-error-code=override

from typing import TYPE_CHECKING
import numpy as np

import foxes.variables as FV
from foxes.core import PointDataModel

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class SetAmbPointResults(PointDataModel):
    """
    This model copies point results to ambient results.

    Attributes
    ----------
    pvars
        The point variables to be treated
    vars
        The variables to be copied to output


    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        super().__init__()
        self.pvars: list[str] = []
        self.vars: list[str] = []

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            It contains coordinate data, model variables, and additional data.
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            It contains coordinate data, model variables, and additional data.

        """
        self.pvars = algo.states.output_point_vars(algo)
        self.vars = [v for v in self.pvars if v in FV.var2amb]
        return super().initialize(algo, loaded_data, force, verbosity)

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
        return [FV.var2amb[v] for v in self.vars] + [FV.WEIGHT]

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
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
            The point data

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values with shape (n_states, n_points)

        """
        ovars = self.output_point_vars(algo)
        for v in self.vars:
            tdata.add(FV.var2amb[v], tdata[v].copy(), tdata.dims[v])
        return {v: tdata[v] for v in ovars}
