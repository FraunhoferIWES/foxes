import dask
import numpy as np
import xarray as xr
from multiprocessing import Pool

from foxes.core import Engine, MData, FData, TData
import foxes.variables as FV
import foxes.constants as FC


class MultiprocessingEngine(Engine):
    """
    The multiprocessing engine for foxes calculations.
    
    Parameters
    ----------
    n_procs: int
        The number of processes to be used,
        or None for automatic
            
    :group: engines
    
    """
    def __init__(
        self, 
        n_procs=None,
        **kwargs,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        n_procs: int, optional
            The number of processes to be used,
            or None for automatic
        kwargs: dict, optional
            Additional parameters for the base class
            
        """
        super().__init__(**kwargs)
        self.n_procs = n_procs
        self._pool = None

    def initialize(self):
        """
        Initializes the engine.
        """      
        self._pool = Pool(processes=self.n_procs)
        super().initialize()

    def finalize(self):
        """
        Finalizes the engine.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None
            
        dask.config.refresh()
        
        super().finalize()