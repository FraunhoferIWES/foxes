class SequentialPlugin:
    """
    Base class for plugins that are
    updated with each sequential iteration

    Parameters
    ----------
    algo: foxes.algorithms.sequential.Sequential
        The sequetial algorithm

    :group: algorithms.sequential.models

    """

    def __init__(self):
        """
        Constructor.
        """
        self.algo = None

    def initialize(self, algo):
        """
        Initialize data based on the intial iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The current sequetial algorithm

        """
        self.algo = algo

    def update(self, algo, fres, pres=None):
        """
        Updates data based on current iteration

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The latest sequetial algorithm
        fres: xarray.Dataset
            The latest farm results
        pres: xarray.Dataset, optional
            The latest point results

        """
        self.algo = algo

    def finalize(self, algo):
        """
        Finalize data based on the final iterator

        Parameters
        ----------
        algo: foxes.algorithms.sequential.Sequential
            The final sequetial algorithm

        """
        self.algo = None
