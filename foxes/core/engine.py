from abc import ABC, abstractmethod

from foxes.utils import all_subclasses
import foxes.constants as FC

__global_engine_data__ = dict(
    engine=None,
)

def get_engine(error=True):
    """
    Gets the global calculation engine
    
    Parameters
    ----------
    error: bool
        Flag for raising ValueError if no
        engine is found
    
    Returns
    -------
    engine: foxes.core.Engine
        The foxes calculation engine
        
    """
    engine = __global_engine_data__["engine"]
    if error and engine is None:
        raise ValueError("Engine not found.")
    return engine

class Engine(ABC):
    """
    Abstract base clas for foxes calculation engines
    
    Attributes
    ----------
    chunk_size_states: int
        The size of a states chunk
    chunk_size_points: int
        The size of a points chunk
    verbosity: int
        The verbosity level, 0 = silent
            
    :group: core
    
    """
    def __init__(
        self, 
        chunk_size_states=None, 
        chunk_size_points=None, 
        verbosity=1,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        chunk_size_states: int, optional
            The size of a states chunk
        chunk_size_points: int, optional
            The size of a points chunk
        verbosity: int
            The verbosity level, 0 = silent
        
        """
        self.chunk_size_states = chunk_size_states
        self.chunk_size_points = chunk_size_points
        self.verbosity = verbosity
        self.__initialized = False
    
    def __repr__(self):
        s = f"chunk_size_states={self.chunk_size_states}, chunk_size_points={self.chunk_size_points}"
        return f"{type(self).__name__}({s})"
    
    @property
    def initialized(self):
        """
        Initialization flag.

        Returns
        -------
        ini: bool
            True if the model has been initialized.

        """
        return self.__initialized
    
    def initialize(self):
        """
        Initializes the engine.
        """
        if not self.initialized:
            if get_engine(error=False) is not None:
                raise ValueError(f"Cannot initialize engine '{type(self).__name__}', since engine already set to '{type(get_engine()).__name__}'")
            global __global_engine_data__
            __global_engine_data__["engine"] = self
            self.__initialized = True
        
    def finalize(self):
        """
        Finalizes the engine.
        """
        if self.initialized:
            global __global_engine_data__
            __global_engine_data__["engine"] = None
            self.__initialized = False
        
    def __enter__(self):
        if not self.initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.initialized:
            self.finalize()
    
    def print(self, *args, level=1, **kwargs):
        """ Prints based on verbosity """
        if self.verbosity >= level:
            print(*args, **kwargs)
    
    @property
    def loop_dims(self):
        """
        Gets the loop dimensions (possibly chunked)
        
        Returns
        -------
        dims: list of str
            The loop dimensions (possibly chunked)
        
        """
        if self.chunk_size_states is None and self.chunk_size_states is None:
            return []
        elif self.chunk_size_states is None:
            return [FC.TARGET]
        elif self.chunk_size_points is None:
            return [FC.STATE]
        else:
            return [FC.STATE, FC.TARGET]
            
    @abstractmethod
    def run_calculation(
        self, 
        algo,
        model, 
        model_data=None, 
        farm_data=None, 
        point_data=None, 
        out_vars=[],
        **calc_pars,
    ):
        """
        Runs the model calculation
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel
            The model that whose calculate function 
            should be run
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset
            The initial farm data
        point_data: xarray.Dataset
            The initial point data
        out_vars: list of str, optional
            Names of the output variables
        calc_pars: dict, optional
            Additional parameters for the model.calculate()
        
        Returns
        -------
        results: xarray.Dataset
            The model results
            
        """
        if not self.initialized:
            raise ValueError(f"Engine '{type(self).__name__}' not initialized")

    @classmethod
    def new(cls, engine_type, *args, **kwargs):
        """
        Run-time engine factory.

        Parameters
        ----------
        engine_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """

        if engine_type is None:
            return None

        allc = all_subclasses(cls)
        found = engine_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == engine_type:
                    return scls(*args, **kwargs)

        else:
            estr = "engine type '{}' is not defined, available types are \n {}".format(
                engine_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
