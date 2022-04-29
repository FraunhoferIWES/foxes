import numpy as np

import foxes.variables as FV

class Data(dict):

    def __init__(self, data, dims, loop_dims):

        self.update(data)
        self.dims = dims
        self.loop_dims = loop_dims

        self.sizes = {}
        for v, d in data.items():
            
            dim = dims[v]

            # remove axes of size 1, added by dask for extra loop dimensions:
            if len(dim) != len(d.shape):
                for li, l in enumerate(loop_dims):
                    if d.shape[li] == 1 and (len(dim) < li + 1 or dim[li] != l):
                        self[v] = np.squeeze(d, axis=li)

            for ci, c in enumerate(dim):
                if c not in self.sizes:
                    self.sizes[c] = d.shape[ci]
                elif self.sizes[c] != d.shape[ci]:
                    raise ValueError(f"Inconsistent size for data entry '{v}', dimension '{c}': Expecting {self.sizes[c]}, found {d.shape[ci]} in shape {d.shape}")

        if FV.STATE in self.sizes:
            self.n_states = self.sizes[FV.STATE]
        if FV.TURBINE in self.sizes:
            self.n_turbines = self.sizes[FV.TURBINE]
        if FV.POINT in self.sizes:
            self.n_points = self.sizes[FV.POINT]
