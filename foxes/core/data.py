import numpy as np

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC


class Data(Dict):
    """
    Container for data and meta data.

    Used during the calculation of single chunks,
    usually for numpy data (not xarray data).

    Parameters
    ----------
    name : str
        The data container name
    data : dict
        The initial data to be stored
    dims : dict
        The dimensions tuples, same or subset
        of data keys
    loop_dims : array_like of str
        List of the loop dimensions during xarray's
        `apply_ufunc` calculations

    Attributes
    ----------
    dims : dict
        The dimensions tuples, same or subset
        of data keys
    loop_dims : array_like of str
        List of the loop dimensions during xarray's
        `apply_ufunc` calculations
    sizes : dict
        The dimension sizes

    """

    def __init__(self, data, dims, loop_dims):
        super().__init__(name="data")

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
                    self.sizes[c] = self[v].shape[ci]
                elif self.sizes[c] != self[v].shape[ci]:
                    raise ValueError(
                        f"Inconsistent size for data entry '{v}', dimension '{c}': Expecting {self.sizes[c]}, found {self[v].shape[ci]} in shape {self[v].shape}"
                    )

        if FV.STATE in self.sizes:
            self.n_states = self.sizes[FV.STATE]
        if FV.TURBINE in self.sizes:
            self.n_turbines = self.sizes[FV.TURBINE]
        if FV.POINT in self.sizes:
            self.n_points = self.sizes[FV.POINT]

        if (
            FV.X in data
            and FV.Y in data
            and FV.H in data
            and dims[FV.X] == (FV.STATE, FV.TURBINE)
            and dims[FV.Y] == (FV.STATE, FV.TURBINE)
            and dims[FV.H] == (FV.STATE, FV.TURBINE)
        ):

            self[FV.TXYH] = np.zeros(
                (self.n_states, self.n_turbines, 3), dtype=FC.DTYPE
            )

            self[FV.TXYH][:, :, 0] = self[FV.X]
            self[FV.TXYH][:, :, 1] = self[FV.Y]
            self[FV.TXYH][:, :, 2] = self[FV.H]

            self[FV.X] = self[FV.TXYH][:, :, 0]
            self[FV.Y] = self[FV.TXYH][:, :, 1]
            self[FV.H] = self[FV.TXYH][:, :, 2]
