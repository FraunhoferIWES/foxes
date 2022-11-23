import pandas as pd

from foxes.core.point_data_model import PointDataModel
from foxes.utils import PandasFileHelper
import foxes.constants as FC
import foxes.variables as FV


class SetUniformData(PointDataModel):
    """
    Set uniform data (can be state dependent)

    Parameters
    ----------
    data_source : str or pandas.DataFrame or dict
        Either a file name, or a data frame, both assuming
        state dependent data. Or a dict for state independent
        uniform data (i.e., scalars)
    output_vars : list of str
        The variables to be written
    var2col : dict
        Mapping from variable names to data column names
    pd_read_pars : dict
        pandas file reading parameters

    Attributes
    ----------
    data_source : str or pandas.DataFrame or dict
        Either a file name, or a data frame, both assuming
        state dependent data. Or a dict for state independent
        uniform data (i.e., scalars)
    ovars : list of str
        The variables to be written
    var2col : dict
        Mapping from variable names to data column names

    """

    def __init__(
        self,
        data_source,
        output_vars,
        var2col={},
        pd_read_pars={},
    ):
        self.data_source = data_source
        self.ovars = output_vars

        self._rpars = pd_read_pars
        self._data = None

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        if self._data is None:
            if isinstance(self.data_source, pd.DataFrame):
                self._data = self.data_source[
                    [self.var2col.get(v, v) for v in self.ovars]
                ].to_numpy(FC.DTYPE)
            elif isinstance(self.data_source, dict):
                pass
            else:
                if verbosity:
                    print(f"States '{self.name}': Reading file {self.data_source}")
                rpars = dict(index_col=0)
                rpars.update(self._rpars)
                self._data = PandasFileHelper().read_file(self.data_source, **rpars)
                self._data = self._data[
                    [self.var2col.get(v, v) for v in self.ovars]
                ].to_numpy(FC.DTYPE)

        super().initialize(algo, verbosity)

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        idata = super().model_input_data(algo)

        if self._data is not None:
            self.VARS = self.var("VARS")
            self.DATA = self.var("DATA")
            idata["coords"][self.VARS] = self.ovars
            idata["data_vars"][self.DATA] = ((FV.STATE, self.VARS), self._data)

        return idata

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
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
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        pdata : foxes.core.Data
            The point data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        for v in self.ovars:
            if self.DATA in mdata:
                pdata[v][:] = mdata[v][None, self.ovars.index(v)]
            else:
                pdata[v][:] = self.data_source[v]

        return {v: pdata[v] for v in self.ovars}

    def finalize(self, algo, results, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level

        """
        if clear_mem:
            self._data = None
        super().finalize(algo, results, clear_mem, verbosity)
