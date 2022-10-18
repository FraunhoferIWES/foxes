import numpy as np

from foxes.core.states import States
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
    org_inds : array_like
        The original states indices, if specified
        in model_input_data

    """

    def __init__(self, states, n_pop):
        super().__init__()
        self.states = states
        self.n_pop = n_pop
        self.org_inds = None

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
        weights = np.zeros((self.n_pop, self.states.size(), algo.n_turbines), dtype=FC.DTYPE)
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
                self.org_inds = coord

        for dname, (dims0, data0) in idata0["data_vars"].items():
            if dims0[0] == FV.STATE:
                shp0 = list(data0.shape)
                shp1 = [self.n_pop] + shp0
                shp2 = [self.size()] + shp0[1:]
                idata["data_vars"][dname] = np.zeros(shp1, dtype=FC.DTYPE)
                idata["data_vars"][dname][:] = data0[None, :]
                idata["data_vars"][dname] = (dims0, idata["data_vars"][dname].reshape(shp2))
            else:
                idata["data_vars"][dname] = (dims0, data0)

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
        return self.states.calculate(algo, mdata, fdata, pdata)

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
