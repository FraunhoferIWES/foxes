import numpy as np
from copy import deepcopy

import foxes.variables as FV
from foxes.core import PointDataModel

class PointWakesCalculation(PointDataModel):

    def __init__(self, point_vars):
        super().__init__()
        self.pvars = point_vars

    def output_farm_vars(self, algo):
        return self.pvars

    def calculate(self, algo, mdata, fdata, pdata):
        
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

                algo.farm_controller.calculate(algo, mdata, fdata, st_sel=trbs)

            if oi < n_order - 1:
                self.pwakes.contribute_to_wake_deltas(algo, mdata, fdata, o, wdeltas)

