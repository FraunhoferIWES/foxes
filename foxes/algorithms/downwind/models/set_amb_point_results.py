
import foxes.variables as FV
from foxes.core import PointDataModel

class SetAmbPointResults(PointDataModel):

    def __init__(self, point_vars, vars_to_amb=None):
        super().__init__()
        self.pvars = point_vars
        self.vars  = vars_to_amb

    def output_point_vars(self, algo):
        if self.pvars is None:
            self.pvars = algo.states.output_point_vars(algo)
        if self.vars is None:
            self.vars = [v for v in self.pvars if v in FV.var2amb]
        return [FV.var2amb[v] for v in self.vars]

    def calculate(self, algo, mdata, fdata, pdata):
        for v in self.vars:
            pdata[FV.var2amb[v]] = pdata[v].copy()
        return {v: pdata[v] for v in self.output_point_vars(algo)}
