class SequentialIterPlugin:
    """
    Base class for plugins that are
    updated with each sequential iteration

    Parameters
    ----------
    iter: foxes.algorithms.sequential.models.SequentialIter
        The current iterator
    
    :group: algorithms.sequential.models

    """

    def __init__(self):
        """
        Constructor.
        """
        self.iter = None

    def initialize(self, iter):
        """
        Initialize data based on the intial iterator
        
        Parameters
        ----------
        iter: foxes.algorithms.sequential.models.SequentialIter
            The initialized iterator
        
        """
        self.iter = iter

    def update(self, iter, fres, pres=None):
        """
        Updates data based on current iteration
        
        Parameters
        ----------
        iter: foxes.algorithms.sequential.models.SequentialIter
            The latest iterator
        fres: xarray.Dataset
            The current farm results
        pres: xarray.Dataset, optional
            The current point results
        
        """
        self.iter = iter
    
    def finalize(self, iter):
        """
        Finalize data based on the final iterator
        
        Parameters
        ----------
        iter: foxes.algorithms.sequential.models.SequentialIter
            The final iterator
        
        """
        self.iter = None
