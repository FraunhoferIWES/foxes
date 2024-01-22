import foxes.variables as FV
import foxes.constants as FC

round_defaults = {v: 4 for v in FV.__dict__.keys() if isinstance(v, str)}
round_defaults[FV.WD] = 3
round_defaults[FV.YAW] = 3
round_defaults[FV.TI] = 6
round_defaults[FV.RHO] = 6
round_defaults[FC.XYH] = 3
round_defaults.update({FV.var2amb[v]: round_defaults[v] for v in FV.var2amb.keys()})
