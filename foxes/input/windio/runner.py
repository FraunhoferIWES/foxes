from foxes.core import Algorithm
from foxes.output import Output

class WindioRunner:
    """
    Runner for windio input
    
    Attributes
    ----------
    algo: foxes.core.Algorithm
        The algorithm object
    output_dicts: list of dict
        The output dictionaries
    farm_results: xarray.Dataset
        The farm results
    output_results: list
        The output results
    verbosity: int
        The verbosity level, 0 = silent

    :group: input.windio

    """
    def __init__(self, algo_dict, output_dicts=[], verbosity=1):
        """
        Conbstructor
        
        Parameters
        ----------
        algo_dict: dict
            The algorithm dictionary
        output_dicts: list of dict
            The output dictionaries
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.algo = algo_dict
        self.output_dicts = output_dicts
        self.verbosity = verbosity
        self.farm_results = None
        self.output_results = None

        self.__initialized = False
    
    def print(self, *args, **kwargs):
        """ Print based on verbosity """
        if self.verbosity > 0:
            print(*args, **kwargs)

    def initialize(self):
        """ Initializes the runner """
        if isinstance(self.algo, dict):
            self.print(f"Creating algorithm '{self.algo['algo_type']}'")
            self.algo = Algorithm.new(**self.algo)
        if not self.algo.initialized:
            self.algo.initialize()
        self.__initialized = True
    
    @property
    def initialized(self):
        """ Flag for initialization """
        return self.__initialized
    
    def run_farm_calc(self):
        """ Runs the farm calculation """
        if not self.__initialized:
            self.initialize()
        self.farm_results = self.algo.calc_farm()
    
    def run_outputs(self):
        """ Runs the output calculation """
        self.output_results = []
        for odict in self.output_dicts:
            self.print("Running output:", odict["output_type"])
            run_fname = odict.pop("run_func")
            run_args = odict.pop("run_args", ())
            run_kwargs = odict.pop("run_kwargs", {})
            o = Output.new(**odict)
            f = getattr(o, run_fname)
            self.output_results.append(f(*run_args, **run_kwargs))
    
    def run(self):
        """ Runs all calculations """
        self.run_farm_calc()
        self.run_outputs()

    def finalize(self):
        """ Initializes the runner """
        if self.algo.initialized:
            self.algo.finalize(clear_mem=True)
        self.algo = None
        self.farm_results = None
        self.output_results = None
        self.__initialized = False
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        self.finalize()
