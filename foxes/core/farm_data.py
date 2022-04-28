import foxes.variables as FV

class FarmData(dict):

    def __init__(self, data, dims, n_turbines):
        self.update(data)
        self.dims = dims

        self.n_states   = len(data[FV.STATE])
        self.n_turbines = n_turbines
