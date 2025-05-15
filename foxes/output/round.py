import foxes.variables as FV

round_defaults = {v: 4 for v in FV.__dict__.keys() if isinstance(v, str)}
round_defaults.update({
    FV.WD: 3,
    FV.WS: 4,
    FV.TI: 6,
    FV.RHO: 5,
    FV.P: 3,
    FV.CT: 6,
    FV.T: 3,
    FV.YLD: 3,
    FV.CAP: 5,
    FV.EFF: 5,
    FV.WEIBULL_A: 3,
    FV.WEIBULL_k: 3,
})
round_defaults.update({FV.var2amb[v]: round_defaults[v] for v in FV.var2amb.keys()})
