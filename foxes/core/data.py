import numpy as np

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC


class Data(Dict):
    """
    Container for data and meta data.

    Used during the calculation of single chunks,
    usually for numpy data (not xarray data).

    Attributes
    ----------
    dims: dict
        The dimensions tuples, same or subset
        of data keys
    loop_dims: array_like of str
        List of the loop dimensions during xarray's
        `apply_ufunc` calculations
    sizes: dict
        The dimension sizes

    :group: core

    """

    def __init__(self, data, dims, loop_dims, name="data"):
        """
        Constructor.

        Parameters
        ----------
        data: dict
            The initial data to be stored
        dims: dict
            The dimensions tuples, same or subset
            of data keys
        loop_dims: array_like of str
            List of the loop dimensions during xarray's
            `apply_ufunc` calculations
        name: str
            The data container name

        """
        super().__init__(name=name)

        self.update(data)
        self.dims = dims
        self.loop_dims = loop_dims

        self.sizes = {}
        for v, d in data.items():
            self.__run_entry_checks(v, d, dims[v])

        self.__auto_update()

    @property
    def n_states(self):
        """
        The number of states

        Returns
        -------
        int:
            The number of states

        """
        return self.sizes[FC.STATE] if FC.STATE in self.sizes else None

    @property
    def n_turbines(self):
        """
        The number of turbines

        Returns
        -------
        int:
            The number of turbines

        """
        return self.sizes[FC.TURBINE] if FC.TURBINE in self.sizes else None

    @property
    def n_points(self):
        """
        The number of points

        Returns
        -------
        int:
            The number of points

        """
        return self.sizes[FC.POINT] if FC.POINT in self.sizes else None

    def states_i0(self, counter=False, algo=None):
        """
        Get the state counter for first state in chunk

        Parameters
        ----------
        counter: bool
            Return the state counter instead of the index
        algo: foxes.core.Algorithm, optional
            The algorithm, required for state counter

        Returns
        -------
        int:
            The state counter for first state in chunk
            or the corresponding index

        """
        if FC.STATE not in self:
            return None
        elif counter:
            if algo is None:
                raise KeyError(f"{self.name}: Missing algo for deducing state counter")
            return np.argwhere(algo.states.index() == self[FC.STATE][0])[0][0]
        else:
            return self[FC.STATE][0]

    def __auto_update(self):
        data = self
        dims = self.dims

        if (
            FV.TXYH not in data
            and FV.X in data
            and FV.Y in data
            and FV.H in data
            and dims[FV.X] == (FC.STATE, FC.TURBINE)
            and dims[FV.Y] == (FC.STATE, FC.TURBINE)
            and dims[FV.H] == (FC.STATE, FC.TURBINE)
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

            self.dims[FV.TXYH] = (FC.STATE, FC.TURBINE, FC.XYH)

    def __run_entry_checks(self, name, data, dims):
        # remove axes of size 1, added by dask for extra loop dimensions:
        if dims is not None:
            if len(dims) != len(data.shape):
                for li, l in enumerate(self.loop_dims):
                    if data.shape[li] == 1 and (len(dims) < li + 1 or dims[li] != l):
                        self[name] = np.squeeze(data, axis=li)

            for ci, c in enumerate(dims):
                if c not in self.sizes:
                    self.sizes[c] = self[name].shape[ci]
                elif self.sizes[c] != self[name].shape[ci]:
                    raise ValueError(
                        f"Inconsistent size for data entry '{name}', dimension '{c}': Expecting {self.sizes[c]}, found {self[name].shape[ci]} in shape {self[name].shape}"
                    )

    def add(self, name, data, dims):
        """
        Add data entry

        Parameters
        ----------
        name: str
            The data name
        data: np.ndarray
            The data
        dims: tuple of str
            The dimensions

        """
        self[name] = data
        self.dims[name] = dims
        self.__run_entry_checks(name, data, dims)
        self.__auto_update()

    @classmethod
    def from_points(
        cls,
        points,
        data={},
        dims={},
        name="pdata",
    ):
        """
        Create from points

        Parameters
        ----------
        points: np.ndarray
            The points, shape: (n_states, n_points, 3)
        data: dict
            The initial data to be stored
        dims: dict
            The dimensions tuples, same or subset
            of data keys
        name: str
            The data container name

        Returns
        -------
        pdata: Data
            The data object

        """
        if len(points.shape) != 3 or points.shape[2] != 3:
            raise ValueError(
                f"Expecting points shape (n_states, n_points, 3), got {points.shape}"
            )
        data[FC.POINTS] = points
        dims[FC.POINTS] = (FC.STATE, FC.POINT, FC.XYH)
        return Data(data, dims, [FC.STATE, FC.POINT], name)
