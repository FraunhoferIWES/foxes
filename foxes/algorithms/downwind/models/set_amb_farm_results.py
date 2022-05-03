
import foxes.variables as FV
from foxes.core import FarmDataModel

class SetAmbFarmResults(FarmDataModel):

    def __init__(self, vars_to_amb=None):
        super().__init__()
        self.vars = vars_to_amb

    def output_farm_vars(self, algo):
        if self.vars is None:
            self.vars = [v for v in algo.farm_vars if v in FV.var2amb]
        return [FV.var2amb[v] for v in self.vars]

    def calculate(self, algo, mdata, fdata):
        for v in self.vars:
            fdata[FV.var2amb[v]] = fdata[v].copy()
        return {v: fdata[v] for v in self.output_farm_vars(algo)}
