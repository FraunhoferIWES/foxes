from abc import abstractmethod

from .point_data_model import PointDataModel
import foxes.variables as FV


class States(PointDataModel):
    """
    Abstract base class for states.

    States describe ambient meteorological data,
    typically wind speed, wind direction, turbulence
    intensity and air density.

    """

    @abstractmethod
    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        pass

    def index(self):
        """
        The index list

        Returns
        -------
        indices : array_like
            The index labels of states, or None for default integers

        """
        pass

    @abstractmethod
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
        pass

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
        idata = {"coords": {}, "data_vars": {}}

        sinds = self.index()
        if sinds is not None:
            idata["coords"][FV.STATE] = sinds

        weights = self.weights(algo)
        if len(weights.shape) != 2:
            raise ValueError(
                f"States '{self.name}': Wrong weights dimension, expecing ({FV.STATE}, {FV.TURBINE}), got shape {weights.shape}"
            )
        if weights.shape[1] != algo.n_turbines:
            raise ValueError(
                f"States '{self.name}': Wrong size of second axis dimension '{FV.TURBINE}': Expecting {algo.n_turbines}, got {weights.shape[1]}"
            )
        idata["data_vars"][FV.WEIGHT] = ((FV.STATE, FV.TURBINE), weights)

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
        return [FV.WS, FV.WD, FV.TI, FV.RHO]
