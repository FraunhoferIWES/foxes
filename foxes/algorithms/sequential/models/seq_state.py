from foxes.core.states import States


class SeqState(States):
    """
    A single state during sequential iteration, just serving
    as a structural placeholder

    Parameters
    ----------
    states: foxes.core.States
        The original states set

    :group: algorithms.sequential.models

    """

    def __init__(self, states):
        """
        Constructor.

        Attributes
        ----------
        states: foxes.core.States
            The original states set

        """
        super().__init__()
        self.states = states

        # updated by SequentialIter:
        self._size = states.size()
        self._indx = None
        self._counter = None

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.states]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        super().initialize(algo, verbosity, force)
        self._size = self.states.size()

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._size

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return [self._indx] if self._size == 1 else self.states.index()

    @property
    def counter(self):
        """
        The current index counter

        Returns
        -------
        i: int
            The current index counter

        """
        return self._counter

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

    def calculate(self, algo, mdata, fdata, tdata, calc_weights=False):
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
        calc_weights: bool
            Flag for weights calculation at points,
            add them to tdata

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        return self.states.calculate(
            algo, mdata, fdata, tdata, calc_weights=calc_weights)
