import foxes.variables as FV

class PointData(dict):

    def __init__(self, data, dims):
        self.update(data)
        self.dims = dims

        self.n_states = data[FV.POINTS].shape[0]
        self.n_points = data[FV.POINTS].shape[1]
