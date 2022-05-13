import numpy as np
import numbers

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC

class LinearWakeSuperposition(WakeSuperposition):

    def __init__(self, scalings):
        super().__init__()
        self.scalings = scalings

    def calc_wakes_plus_wake(self, algo, mdata, fdata, states_source_turbine,
                                sel_sp, variable, wake_delta, wake_model_result):

        if isinstance(self.scalings, dict):
            try:
                scaling = self.scalings[variable]
            except KeyError:
                raise KeyError(f"Model '{self.name}': No scaling found for wake variable '{variable}'")
        else:
            scaling = self.scalings

        if scaling is None:
            wake_delta[sel_sp] += wake_model_result
            return wake_delta
        
        elif isinstance(scaling, numbers.Number):
            wake_delta[sel_sp] += scaling * wake_model_result
            return wake_delta

        elif len(scaling) >= 14 and (
                scaling == f'source_turbine' \
                or scaling == 'source_turbine_amb' \
                or (len(scaling) > 15 and scaling[14] == '_')
            ):

            if scaling == f'source_turbine':
                var = variable
            elif scaling == 'source_turbine_amb':
                var = FV.var2amb[variable]
            else:
                var = scaling[15:]

            try:
                vdata = fdata[var]
                
            except KeyError:
                raise KeyError(f"Model '{self.name}': Scaling variable '{var}' for wake variable '{variable}' not found in fdata {sorted(list(fdata.keys()))}")
            
            n_states = mdata.n_states
            n_points = wake_delta.shape[1]
            stsel    = (np.arange(n_states), states_source_turbine)
            scale    = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            scale[:] = vdata[stsel][:, None]
            scale    = scale[sel_sp]

            wake_delta[sel_sp] += scale * wake_model_result

            return wake_delta
        
        else:
            raise ValueError(f"Model '{self.name}': Invalid scaling choice '{scaling}' for wake variable '{variable}', valid choices: None, <scalar>, 'source_turbine', 'source_turbine_<var>'")

    def calc_final_wake_delta(self, algo, mdata, fdata, variable, amb_results, wake_delta):
        return wake_delta
        
