import numpy as np
from copy import deepcopy

import foxes.variables as FV
from foxes.core import FarmDataModel

class FarmWakesCalculation(FarmDataModel):

    def output_farm_vars(self, algo):
        ovars  = deepcopy(algo.rotor_model.output_farm_vars(algo))
        ovars += algo.farm_controller.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def initialize(self, algo):
        super().initialize(algo)
        self.pwakes = algo.partial_wakes_model
        if not self.pwakes.initialized:
            self.pwakes.initialize(algo)

    def calculate(self, algo, mdata, fdata):
        
        torder   = fdata[FV.ORDER]
        n_order  = torder.shape[1]
        n_states = mdata.n_states

        wdeltas = self.pwakes.new_wake_deltas(algo, mdata, fdata)

        for oi in range(n_order):

            o = torder[:, oi]

            if oi > 0:

                self.pwakes.evaluate_results(algo, mdata, fdata, wdeltas, states_turbine=o)

                trbs = np.zeros((n_states, algo.n_turbines), dtype=bool)
                np.put_along_axis(trbs, o[:, None], True, axis=1)

                res = algo.farm_controller.calculate(algo, mdata, fdata, pre_rotor=False, st_sel=trbs)
                fdata.update(res)

            if oi < n_order - 1:
                self.pwakes.contribute_to_wake_deltas(algo, mdata, fdata, o, wdeltas)

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
