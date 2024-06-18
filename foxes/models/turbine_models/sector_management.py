import numpy as np
import pandas as pd

from foxes.core import TurbineModel
from foxes.utils import PandasFileHelper
import foxes.variables as FV
import foxes.constants as FC


class SectorManagement(TurbineModel):
    """
    Changes variables based on variable range conditions.

    Attributes
    ----------
    source: str or pandas.DataFrame
        The file path or data

    :group: models.turbine_models

    """

    def __init__(
        self,
        data_source,
        range_vars,
        target_vars,
        col_tinds=None,
        col_tnames=None,
        colmap={},
        var_periods={FV.WD: 360.0, FV.AMB_WD: 360.0},
        pd_file_read_pars={},
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            The file path or data
        range_vars: list of str
            The variables for which (min, max) ranges
            are specified in the data
        target_vars: list of str
            The variables that change if range variables
            are within specified ranges
        col_tinds: str, optional
            The turbine index column name in the data
        col_tnames: str, optional
            The turbine name column name in the data
        colmap: dict
            Mapping from expected to existing
            column names
        var_periods: dict
            Periods for periodic variables
        pd_file_read_pars: dict
            Parameters for pandas file reading

        """
        super().__init__()

        self.source = data_source

        self._col_i = col_tinds
        self._col_t = col_tnames
        self._rvars = range_vars
        self._tvars = target_vars
        self._colmap = colmap
        self._perds = var_periods
        self._rpars = pd_file_read_pars

        self._rdata = None
        self._tdata = None
        self._trbs = None

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(algo, verbosity)

        if isinstance(self.source, pd.DataFrame):
            data = self.source
        else:
            if verbosity > 0:
                print(f"{self.name}: Reading file {self.source}")
            data = PandasFileHelper.read_file(self.source, **self._rpars)

        if self._trbs is None:
            if self._col_i is not None and self._col_t is None:
                data.reset_index(inplace=True)
            elif self._col_i is None and self._col_t is not None:
                tnames = algo.farm.turbine_names
                inds = [tnames.index(name) for name in data[self._col_t]]
                data[FC.TURBINE] = inds
                self._col_i = FC.TURBINE
            else:
                raise KeyError(
                    f"{self.name}: Please either specify 'col_tinds' or 'col_tnames'"
                )
            self._trbs = data[self._col_i].to_numpy()
        n_trbs = len(self._trbs)

        self._rcols = []
        for v in self._rvars:
            col_vmin = f"{v}_min"
            col_vmin = self._colmap.get(col_vmin, col_vmin)
            if col_vmin not in data.columns:
                raise KeyError(
                    f"{self.name}: Missing column '{col_vmin}', maybe add it to 'colmap'?"
                )

            col_vmax = f"{v}_max"
            col_vmax = self._colmap.get(col_vmax, col_vmax)
            if col_vmax not in data.columns:
                raise KeyError(
                    f"{self.name}: Missing column '{col_vmax}', maybe add it to 'colmap'?"
                )

            self._rcols += [col_vmin, col_vmax]

        self._tcols = []
        for v in self._tvars:
            col = self._colmap.get(v, v)
            if col not in data.columns:
                raise KeyError(
                    f"{self.name}: Missing column '{col}', maybe add it to 'colmap'?"
                )
            self._tcols.append(col)

        n_rvars = len(self._rvars)
        self._rdata = data[self._rcols].to_numpy().reshape(n_trbs, n_rvars, 2)
        self._tdata = data[self._tcols].to_numpy()

        for vi, v in enumerate(self._rvars):
            if v in self._perds:
                self._rdata[:, vi] = np.mod(self._rdata[:, vi], self._perds[v])

    def output_farm_vars(self, algo):
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
        return self._tvars

    def calculate(self, algo, mdata, fdata, st_sel):
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
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        # prepare:
        n_trbs = len(self._trbs)
        if n_trbs == fdata.n_turbines and np.all(
            self._trbs == np.arange(fdata.n_turbines)
        ):
            tsel = np.s_[:]
        else:
            tsel = self._trbs

        # find state-turbine data that matches ranges:
        rsel = np.ones((fdata.n_states, n_trbs), dtype=bool)
        for vi, v in enumerate(self._rvars):
            d = fdata[v][:, tsel]
            if v in self._perds:
                d = np.mod(d, self._perds[v])
                mi = self._rdata[:, vi, 0]
                ma = self._rdata[:, vi, 1]
                sel = ma < mi
                if np.any(sel):
                    rsel[:, sel] = rsel[:, sel] & (
                        (d[:, sel] >= mi[sel]) | (d[:, sel] < ma[sel])
                    )
                if np.any(~sel):
                    rsel[:, ~sel] = (
                        rsel[:, ~sel]
                        & (d[:, ~sel] >= mi[~sel])
                        & (d[:, ~sel] < ma[~sel])
                    )
            else:
                rsel = (
                    rsel
                    & (d >= self._rdata[None, :, vi, 0])
                    & (d < self._rdata[None, :, vi, 1])
                )

        # set target data:
        if np.any(rsel):
            sel = np.where(rsel)
            selt = self._trbs[sel[1]]
            for vi, v in enumerate(self._tvars):
                fdata[v][sel[0], selt] = self._tdata[None, sel[1], vi]

        return {v: fdata[v] for v in self._tvars}
