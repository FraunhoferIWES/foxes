import numpy as np

from foxes.core import States, Data
import foxes.constants as FC
import foxes.variables as FV


class PopStates(States):
    """
    Helper class for vectorized opt population
    calculation, via artificial states of length
    n_pop times n_states.

    Parameters
    ----------
    states : foxes.core.States
        The original states
    n_pop : int
        The population size

    Attributes
    ----------
    states : foxes.core.States
        The original states
    n_pop : int
        The population size

    """

    def __init__(self, states, n_pop):
        super().__init__()
        self.states = states
        self.n_pop = n_pop

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
        if not self.states.initialized:
            self.states.initialize(algo, verbosity=verbosity)
        super().initialize(algo, verbosity)

        self.STATE0 = self.var(FV.STATE + "0")
        self.SMAP = self.var("SMAP")

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size() * self.n_pop

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights : numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        weights = np.zeros(
            (self.n_pop, self.states.size(), algo.n_turbines), dtype=FC.DTYPE
        )
        weights[:] = self.states.weights(algo)[None, :, :] / self.n_pop
        return weights.reshape(self.size(), algo.n_turbines)

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function is automatically called during
        initialization. It should specify all data
        that is either state or point dependent, or
        intended to be shared between chunks.

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
        idata0 = self.states.model_input_data(algo)
        idata = super().model_input_data(algo)

        for cname, coord in idata0["coords"].items():
            if cname != FV.STATE:
                idata["coords"][cname] = coord
            else:
                idata["coords"][self.STATE0] = coord

        for dname, (dims0, data0) in idata0["data_vars"].items():
            if dname != FV.WEIGHT:
                hdims = tuple([d if d != FV.STATE else self.STATE0 for d in dims0])
                idata["data_vars"][dname] = (hdims, data0)

        smap = np.zeros((self.n_pop, self.states.size()), dtype=np.int32)
        smap[:] = np.arange(self.states.size())[None, :]
        smap = smap.reshape(self.size())
        idata["data_vars"][self.SMAP] = ((FV.STATE,), smap)

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
        return self.states.output_point_vars(algo)

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

        hdata = {}
        hdims = {}
        smap = mdata[self.SMAP]
        for dname, data in mdata.items():
            dms = mdata.dims[dname]
            if dname == self.SMAP or dname == self.STATE0:
                pass
            elif dms[0] == self.STATE0:
                hdata[dname] = data[smap]
                hdims[dname] = tuple([FV.STATE] + list(dms)[1:])
            elif self.STATE0 in dms:
                raise ValueError(
                    f"States '{self.name}': Found states variable not at dimension 0 for mdata entry '{dname}': {dms}"
                )
            else:
                hdata[dname] = data
                hdims[dname] = dms
        hmdata = Data(hdata, hdims, mdata.loop_dims)

        return self.states.calculate(algo, hmdata, fdata, pdata)

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
        self.states.finalize(algo, results, clear_mem, verbosity)
        super().finalize(algo, results, clear_mem, verbosity)
