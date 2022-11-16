import numpy as np
import pandas as pd

from foxes.core import TurbineModel
from foxes.utils import PandasFileHelper
import foxes.variables as FV
import foxes.constants as FC


class SectorManagement(TurbineModel):
    """
    Changes variables based on variable range conditions.

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        The file path or data
    range_vars : list of str
        The variables for which (min, max) ranges
        are specified in the data
    target_vars : list of str
        The variables that change if range variables
        are within specified ranges
    colmap : dict
        Mapping from expected to existing
        column names
    pd_file_read_pars : dict, optional
        Parameters for pandas file reading

    Attributes
    ----------
    source : str or pandas.DataFrame
        The file path or data

    """

    def __init__(
            self, 
            data_source,
            range_vars,
            target_vars,
            col_index=None,
            col_turbine=None,
            colmap={},
            pd_file_read_pars={},
        ):
        super().__init__()

        self.source = data_source

        self._col_i = col_index
        self._col_t = col_turbine
        self._rvars = range_vars
        self._tvars = target_vars
        self._colmap = colmap
        self._rpars = pd_file_read_pars

        self._rdata = None
        self._tdata = None

    def initialize(self, algo, st_sel, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)
        verbosity : int
            The verbosity level

        """
        if self._rdata is None or self._tdata is None:

            if isinstance(self.source, pd.DataFrame):
                data = self.source
            else:
                if verbosity > 0:
                    print(f"{self.name}: Reading file {self.source}")
                data = PandasFileHelper.read_file(self.source, **self._rpars)
            
            data.reset_index(inplace=True)
            if self._col_i is not None and self._col_t is None:
                data.set_index(self._col_i, inplace=True)
            elif self._col_i is None and self._col_t is not None:
                tnames = algo.farm.turbine_names()
                inds = [tnames.index(name) for name in data[self._col_t]]
                data[FV.turbine] = inds
                data.set_index(FV.TURBINE, inplace=True)
            else:
                raise KeyError(f"{self.name}: Please either specify 'col_index' or 'col_turbine'")
            data.sort_index(inplace=True)
            self._turbines = data.index.to_numpy()

            self._rcols = []
            for v in self._rvars:

                col_vmin = f"{v}_min"
                col_vmin = self._colmap.get(col_vmin, col_vmin)
                if col_vmin not in data.columns:
                    raise KeyError(f"{self.name}: Missing column '{col_vmin}', maybe add it to 'colmap'?")

                col_vmax = f"{v}_max"
                col_vmax = self._colmap.get(col_vmax, col_vmax)
                if col_vmax not in data.columns:
                    raise KeyError(f"{self.name}: Missing column '{col_vmax}', maybe add it to 'colmap'?")

                self._rcols += [col_vmin, col_vmax]

            self._tcols = []
            for v in self._tvars:
                col = self._colmap.get(v, v)
                if col not in data.columns:
                    raise KeyError(f"{self.name}: Missing column '{col}', maybe add it to 'colmap'?")
                self._tcols.append(col)

            n_rvars = len(self._rvars)
            self._rdata = data[self._rcols].to_numpy().reshape(n_rvars, 2)
            self._tdata = data[self._tcols].to_numpy()

        super().initialize(algo, st_sel, verbosity)

    def output_farm_vars(self, algo):
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
        return self._tvars

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """

        # prepare:
        n_trbs = len(self._turbines)
        tsel = np.s_[:] if np.all(self._turbines == np.arange(fdata.n_turbines)) else self._turbines

        # find state-turbine data that matches ranges:
        rsel = np.ones((fdata.n_states, n_trbs), dtype=bool)
        for vi, v in enumerate(self._rvars):
            d = fdata[v][:, tsel]
            rsel = rsel & (d >= self._rdata[vi, 0]) & (d < self._rdata[vi, 1]) 

        # set target data:
        if np.any(rsel):
            sel = np.ones((fdata.n_states, algo.n_turbines), dtype=bool)
            sel[: tsel] = rsel
            for vi, v in enumerate(self._tvars):
                fdata[v][sel] = self._tdata[vi]

        return {v: fdata[v] for v in self._tvars}
