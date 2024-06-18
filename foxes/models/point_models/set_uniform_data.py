import pandas as pd

from foxes.core.point_data_model import PointDataModel
from foxes.utils import PandasFileHelper
import foxes.constants as FC
import foxes.variables as FV


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
        data_source,
        output_vars,
        var2col={},
        pd_read_pars={},
    ):
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
        self.data_source = data_source
        self.ovars = output_vars
        self.var2col = var2col

        self._rpars = pd_read_pars

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.VARS = self.var("VARS")
        self.DATA = self.var("DATA")

        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source[
                [self.var2col.get(v, v) for v in self.ovars]
            ].to_numpy(FC.DTYPE)
        elif isinstance(self.data_source, dict):
            pass
        else:
            if verbosity:
                print(f"States '{self.name}': Reading file {self.data_source}")
            rpars = dict(index_col=0)
            rpars.update(self._rpars)
            data = PandasFileHelper().read_file(self.data_source, **rpars)
            data = data[[self.var2col.get(v, v) for v in self.ovars]].to_numpy(FC.DTYPE)

        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = self.ovars
        idata["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data)

        return idata

    def output_point_vars(self, algo):
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

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
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
                pdata[v][:] = mdata[v][None, self.ovars.index(v)]
            else:
                pdata[v][:] = self.data_source[v]

        return {v: pdata[v] for v in self.ovars}
