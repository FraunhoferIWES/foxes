import yaml

from foxes.core import Algorithm
from foxes.output import Output


def _write_yaml(data, fpath):
    """ Write the data to yaml """
    rmap = {
        "include": "!include",
    }
    with open(fpath, "w") as file:
        yaml.dump(data, file)
    with open(fpath, "r") as f:
        s = f.read()
    with open(fpath, 'w') as f:
        for k1, k2 in rmap.items():
            s = s.replace(k1, k2)
        f.write(s)
        
class WindioRunner:
    """
    Runner for windio input

    Attributes
    ----------
    algo: foxes.core.Algorithm
        The algorithm object
    output_dir: pathlib.Path
        Path to the output folder
    output_dicts: list of dict
        The output dictionaries
    farm_results: xarray.Dataset
        The farm results
    output_results: list
        The output results
    wio_input_data: dict
        The wind_energy_system windio input data
    file_name_input_yaml: str
        Name of the written input data file
    file_name_output_yaml: str
        Name of the written output data file
    verbosity: int
        The verbosity level, 0 = silent

    :group: input.windio

    """

    def __init__(
        self, 
        algo_dict, 
        output_dir=".",
        output_dicts=[], 
        wio_input_data=None, 
        file_name_input_yaml="recorded_input.yaml",
        file_name_output_yaml="recorded_output.yaml",
        verbosity=1,
        ):
        """
        Conbstructor

        Parameters
        ----------
        algo_dict: dict
            The algorithm dictionary
        output_dir: pathlib.Path
            Path to the output folder
        output_dicts: list of dict
            The output dictionaries
        wio_input_data: dict
            The wind_energy_system windio input data
        file_name_input_yaml: str
            Name of the written input data file
        file_name_output_yaml: str
            Name of the written output data file
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.algo = algo_dict
        self.output_dir = output_dir
        self.output_dicts = output_dicts
        self.wio_input_data = wio_input_data
        self.file_name_input_yaml = file_name_input_yaml
        self.file_name_output_yaml = file_name_output_yaml
        self.verbosity = verbosity
        self.farm_results = None
        self.output_results = None

        self.__initialized = False
        
        self._output_yaml = {}
        if wio_input_data is not None and len(wio_input_data):
            fpath = output_dir/file_name_input_yaml
            self.print(f"Writing file", fpath)
            _write_yaml(wio_input_data, fpath)
            self._output_yaml["wind_energy_system"] = f"include {file_name_input_yaml}"

    def print(self, *args, level=1, **kwargs):
        """Print based on verbosity"""
        if self.verbosity >= level:
            print(*args, **kwargs)

    def initialize(self):
        """Initializes the runner"""
        if isinstance(self.algo, dict):
            self.print(f"Creating algorithm '{self.algo['algo_type']}'", level=2)
            self.algo = Algorithm.new(**self.algo)
        if not self.algo.initialized:
            self.algo.initialize()
        self.__initialized = True

    @property
    def initialized(self):
        """Flag for initialization"""
        return self.__initialized

    def run_farm_calc(self):
        """Runs the farm calculation"""
        if not self.__initialized:
            self.initialize()
        self.print("Running farm_calc")
        self.farm_results = self.algo.calc_farm()

    def run_outputs(self):
        """Runs the output calculation"""
        self.output_results = []
        for odict in self.output_dicts:
            self.print("Running output:", odict["output_type"])
            run_fname = odict.pop("run_func")
            run_args = odict.pop("run_args", ())
            run_kwargs = odict.pop("run_kwargs", {})
                    
            _odict = odict.copy()
            if "output_yaml_update" in _odict:
                self._output_yaml.update(_odict.pop("output_yaml_update"))
            if _odict.pop("farm_results", False):
                _odict["farm_results"] = self.farm_results
            if _odict.pop("algo", False):
                _odict["algo"] = self.algo
            o = Output.new(**_odict)
            f = getattr(o, run_fname)
            self.output_results.append(f(*run_args, **run_kwargs))
            
        fpath = self.output_dir/self.file_name_output_yaml
        self.print(f"Writing file", fpath)
        _write_yaml(self._output_yaml, fpath)
            
    def run(self):
        """Runs all calculations"""
        self.run_farm_calc()
        self.run_outputs()

    def finalize(self):
        """Initializes the runner"""
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
