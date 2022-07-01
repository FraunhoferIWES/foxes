import foxes.variables as FV
from foxes.core import FarmDataModel


class CalcOrder(FarmDataModel):
    """
    This model calculates the turbine evaluation
    order, via wake frames.
    """

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
        if not algo.wake_frame.initialized:
            if verbosity:
                print(
                    f"{self.name}, linked model '{algo.wake_frame.name}': Initializing"
                )
            algo.wake_frame.initialize(algo, verbosity)

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
        return [FV.ORDER]

    def calculate(self, algo, mdata, fdata):
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

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        return {FV.ORDER: algo.wake_frame.calc_order(algo, mdata, fdata)}
