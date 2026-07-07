from __future__ import annotations
# mypy: disable-error-code=override

import pandas as pd
from typing import TYPE_CHECKING, Any

from foxes.core.point_data_model import PointDataModel
from foxes.utils import PandasFileHelper
from foxes.config import config
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class SetUniformData(PointDataModel):
    """
    Set uniform data (can be state dependent)

    Attributes
    ----------
    data_source: str or pandas.DataFrame or dict
        Either a file name, or a data frame, both assuming
        state dependent data. Or a dict for state independent
        uniform data (i.e., scalars)
    ovars: list of str
        The variables to be written
    var2col: dict
        Mapping from variable names to data column names

    :group: models.point_models

    """

    def __init__(
        self,
        data_source: str | pd.DataFrame | dict[str, Any],
        output_vars: list[str],
        var2col: dict[str, str] = {},
        pd_read_pars: dict[str, Any] = {},
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame or dict
            Either a file name, or a data frame, both assuming
            state dependent data. Or a dict for state independent
            uniform data (i.e., scalars)
        output_vars: list of str
            The variables to be written
        var2col: dict
            Mapping from variable names to data column names
        pd_read_pars: dict
            pandas file reading parameters

        """
        super().__init__()

        self.data_source = data_source
        self.ovars = output_vars
        self.var2col = var2col

        self._rpars = pd_read_pars

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.VARS = self.var("VARS")
        self.DATA = self.var("DATA")

        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source[
                [self.var2col.get(v, v) for v in self.ovars]
            ].to_numpy(config.dtype_double)
        elif isinstance(self.data_source, dict):
            data = None
        else:
            if verbosity:
                print(f"States '{self.name}': Reading file {self.data_source}")
            rpars = dict(index_col=0)
            rpars.update(self._rpars)
            data = PandasFileHelper().read_file(self.data_source, **rpars)
            data = data[[self.var2col.get(v, v) for v in self.ovars]].to_numpy(
                config.dtype_double
            )

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        if data is not None:
            loaded_data["coords"][self.VARS] = self.ovars
            loaded_data["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data)

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return self.ovars

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        pdata: TData,
    ) -> dict[str, Any]:
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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        for v in self.ovars:
            if self.DATA in mdata:
                values = mdata[self.DATA][:, self.ovars.index(v)]
                pdata[v][:] = values[:, None]
            else:
                dsource = self.data_source
                assert isinstance(dsource, dict), (
                    f"{self.name}: Missing loaded data '{self.DATA}' requires dict data_source"
                )
                values = dsource[v]
                if hasattr(values, "ndim") and values.ndim == 1:
                    pdata[v][:] = values[:, None]
                else:
                    pdata[v][:] = values

        return {v: pdata[v] for v in self.ovars}
