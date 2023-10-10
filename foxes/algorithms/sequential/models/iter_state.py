

from foxes.core.states import States

class IterStates(States):
    """
    A states iterator
    
    Parameters
    ----------
    states: foxes.core.States
        The original states set

    """

    def __init__(self, states):
        """
        Constructor.

        Attributes
        ----------
        states: foxes.core.States
            The original states set
        
        """
        self.states = states

    def __iter__(self):
        """ Initialize use as iterator """
        self._inds = self.states.index()
        self._weights = self.states.weights()
        self._si = 0
        return self
    
    def __next__(self):
        """ Evaluate the next state """
        if self._si < len(self._inds):
            self._si += 1
            return self._si, self._inds[self._si], self._weights[self._si]
        else:
            del self._inds, self._weights, self._si
            raise StopIteration

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return [self._inds[self._si]]
    
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
        return self._weights[self._si, None, :]
    
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
        return self.states.output_point_vars(algo)

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        TODO
