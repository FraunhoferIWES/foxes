import numpy as np

from foxes.core import FarmModel

class Turbine2FarmModel(FarmModel):

    def __init__(self, turbine_model):
        super().__init__()
        self.turbine_model = turbine_model

    def __repr__(self):
        return f"{type(self).__name__}({self.turbine_model})"

    def initialize(self, algo, farm_data, **parameters):
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        self.turbine_model.initialize(algo, farm_data, st_sel=s, **parameters)
        super().initialize(algo, farm_data)
    
    def output_farm_vars(self, algo):
        return self.turbine_model.output_farm_vars(algo)
    
    def calculate(self, algo, fdata, **parameters):
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        return self.turbine_model.calculate(algo, fdata, st_sel=s, **parameters)

    def finalize(self, algo, farm_data, **parameters):
        s = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
        self.turbine_model.finalize(algo, farm_data, st_sel=s, **parameters)
    