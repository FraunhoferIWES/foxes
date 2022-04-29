import foxes.variables as FV

class FData(dict):

    def __init__(self, data, dims):
        self.update(data)
        self.dims = dims

        data0 = next(iter(data.values()))
        self.n_states   = data0.shape[0]
        self.n_turbines = data0.shape[1]

class PData(dict):

    def __init__(self, data, dims):
        self.update(data)
        self.dims = dims

        self.n_states = data[FV.POINTS].shape[0]
        self.n_points = data[FV.POINTS].shape[1]

class MData(dict):

    def __init__(self, data, dims):
        self.update(data)
        self.dims = dims

        if FV.STATE in data:
            self.n_states = len(data[FV.STATE])
