import numpy as np

from foxes.core import States
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class ScanStates(States):
    """
    Scan over selected variables

    Parameters
    ----------
    scans: dict
        The scans, key: variable name,
        value: scan values

    :group: input.states

    """

    def __init__(self, scans, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        scans: dict
            The scans, key: variable name,
            value: scan values
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(**kwargs)
        self.scans = {v: np.asarray(d) for v, d in scans.items()}

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
        n_v = len(self.scans)
        shp = [len(v) for v in self.scans.values()]
        self._N = np.prod(shp)
        self._vars = list(self.scans.keys())

        data = np.zeros(shp + [n_v], dtype=config.dtype_double)
        for i, d in enumerate(self.scans.values()):
            s = [None] * n_v
            s[i] = np.s_[:]
            s = tuple(s)
            data[..., i] = d[s]
        data = data.reshape(self._N, n_v)

        self.VARS = self.var("vars")
        self.DATA = self.var("data")
        idata = super().load_data(algo, verbosity)
        idata["coords"][self.VARS] = self._vars
        idata["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data)

        return idata

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        data_stash[self.name].update(dict(scans=self.scans))
        del self.scans

    def unset_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        data = data_stash[self.name]
        self.scans = data.pop("scans")

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

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
        return self._vars

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights: numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        return np.full(
            (self._N, algo.n_turbines), 1.0 / self._N, dtype=config.dtype_double
        )

    def calculate(self, algo, mdata, fdata, tdata):
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
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        for i, v in enumerate(self._vars):
            if v not in tdata:
                tdata[v] = np.zeros_like(tdata[FC.TARGETS][..., 0])
            tdata[v][:] = mdata[self.DATA][:, None, None, i]

        return {v: tdata[v] for v in self.output_point_vars(algo)}
