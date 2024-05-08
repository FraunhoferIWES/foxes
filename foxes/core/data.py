import numpy as np

from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC


class Data(Dict):
    """
    Container for numpy array data and
    the associated meta data.

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
            self._run_entry_checks(v, d, dims[v])

        self._auto_update()

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

    def _auto_update(self):
        """Checks and operations after data changes"""
        data = self
        dims = self.dims

        if (
            FV.TXYH not in data
            and FV.X in data
            and FV.Y in data
            and FV.H in data
            and dims[FV.X] == dims[FV.Y]
            and dims[FV.X] == dims[FV.H]
        ):
            self[FV.TXYH] = np.stack([self[FV.X], self[FV.Y], self[FV.H]], axis=-1)

            self[FV.X] = self[FV.TXYH][..., 0]
            self[FV.Y] = self[FV.TXYH][..., 1]
            self[FV.H] = self[FV.TXYH][..., 2]

            self.dims[FV.TXYH] = tuple(list(dims[FV.X]) + [FC.XYH])

    def _run_entry_checks(self, name, data, dims):
        """Run entry checks on new data"""
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
        self._run_entry_checks(name, data, dims)
        self._auto_update()

    def get_slice(self, s, dim_map={}, name=None, keep=True):
        """
        Get a slice of data.

        Parameters
        ----------
        s: slice
            The slice
        dim_map: dict
            Mapping from original to new dimensions.
            If not found, same dimensions are assumed.
        name: str, optional
            The name of the data object
        keep: bool
            Keep non-matching fields as they are, else
            throw them out

        Returns
        -------
        data: Data
            The new data object, containing slices

        """
        data = {}
        dims = {}
        for v in self.keys():
            try:
                d = self.dims[v]
                data[v] = self[v][s]
                dims[v] = dim_map.get(d, d)
            except IndexError:
                if keep:
                    data[v] = self[v]
                    dims[v] = self.dims[v]
        if name is None:
            name = self.name
        return type(self)(data, dims, loop_dims=self.loop_dims, name=name)


class MData(Data):
    """
    Container for foxes model data.

    :group: core

    """

    def __init__(self, *args, name="mdata", **kwargs):
        """
        Constructor

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        name: str
            The data name
        kwargs: dict, optional
            Arguments for the base class

        """
        super().__init__(*args, name=name, **kwargs)


class FData(Data):
    """
    Container for foxes farm data.

    Each farm data entry has (n_states, n_turbines) shape,
    except the dimensions.

    :group: core

    """

    def __init__(self, *args, name="fdata", **kwargs):
        """
        Constructor

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        name: str
            The data name
        kwargs: dict, optional
            Arguments for the base class

        """
        super().__init__(*args, name=name, **kwargs)

    def _run_entry_checks(self, name, data, dims):
        """Run entry checks on new data"""
        super()._run_entry_checks(name, data, dims)
        data = self[name]
        dims = self.dims[name]

        if name not in self.sizes and name not in FC.TNAME:
            dms = (FC.STATE, FC.TURBINE)
            shp = (self.n_states, self.n_turbines)
            if len(data.shape) < 2:
                raise ValueError(
                    f"FData '{self.name}': Invalid shape for '{name}', expecting {shp}, got {data.shape}"
                )
            if len(dims) < 2 or dims[:2] != dms:
                raise ValueError(
                    f"FData '{self.name}': Invalid dims for '{name}', expecting {dms}, got {dims}"
                )

    def _auto_update(self):
        """Checks and operations after data changes"""
        super()._auto_update()
        if len(self):
            for x in [FC.STATE, FC.TURBINE]:
                if x not in self.sizes:
                    raise KeyError(
                        f"FData '{self.name}': Missing '{x}' in sizes, got {sorted(list(self.sizes.keys()))}"
                    )


class TData(Data):
    """
    Container for foxes target data.

    Each target consists of a fixed number of
    target points.

    :group: core

    """

    def __init__(self, *args, name="tdata", **kwargs):
        """
        Constructor

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        name: str
            The data name
        kwargs: dict, optional
            Arguments for the base class

        """
        super().__init__(*args, name=name, **kwargs)

    def _run_entry_checks(self, name, data, dims):
        """Run entry checks on new data"""
        super()._run_entry_checks(name, data, dims)
        data = self[name]
        dims = self.dims[name]

        if name == FC.TARGETS:
            dms = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
            shp = (self.n_states, self.n_targets, self.n_tpoints, 3)
            if dims != dms:
                raise ValueError(
                    f"TData '{self.name}': Invalid dims of {FC.TARGETS}, expecting {dms}, got {dims}"
                )
            if data.shape != shp:
                raise ValueError(
                    f"TData '{self.name}': Invalid shape of {FC.TARGETS}, expecting {shp}, got {data.shape}"
                )

        elif name == FC.TWEIGHTS:
            dms = (FC.TPOINT,)
            shp = (self.n_tpoints,)
            if dims != dms:
                raise ValueError(
                    f"TData '{self.name}': Invalid dims of {FC.TWEIGHTS}, expecting {dms}, got {dims}"
                )
            if data.shape != shp:
                raise ValueError(
                    f"TData '{self.name}': Invalid shape of {FC.TWEIGHTS}, expecting {shp}, got {data.shape}"
                )

        elif FC.TARGETS not in self:
            raise KeyError(
                f"TData '{self.name}': Missing '{FC.TARGETS}' before adding '{name}'"
            )

        elif FC.TWEIGHTS not in self:
            raise KeyError(
                f"TData '{self.name}': Missing '{FC.TWEIGHTS}' before adding '{name}'"
            )

        elif name not in self.sizes:
            dms = (FC.STATE, FC.TARGET, FC.TPOINT)
            shp = (self.n_states, self.n_targets, self.n_tpoints)
            if len(data.shape) < 3:
                raise ValueError(
                    f"TData '{self.name}': Invalid shape for '{name}', expecting {shp}, got {data.shape}"
                )
            if len(dims) < 3 or dims[:3] != dms:
                raise ValueError(
                    f"TData '{self.name}': Invalid dims for '{name}', expecting {dms}, got {dims}"
                )

    def _auto_update(self):
        """Checks and operations after data changes"""
        super()._auto_update()
        if len(self):
            for x in [FC.TARGETS, FC.TWEIGHTS]:
                if x not in self:
                    raise KeyError(
                        f"TData '{self.name}': Missing '{x}' in data, got {sorted(list(self.keys()))}"
                    )
                if x not in self.dims:
                    raise KeyError(
                        f"TData '{self.name}': Missing '{x}' in dims, got {sorted(list(self.dims.keys()))}"
                    )
            for x in [FC.STATE, FC.TARGET, FC.TPOINT]:
                if x not in self.sizes:
                    raise KeyError(
                        f"TData '{self.name}': Missing '{x}' in sizes, got {sorted(list(self.sizes.keys()))}"
                    )

    @property
    def n_targets(self):
        """
        The number of targets

        Returns
        -------
        int:
            The number of targets

        """
        return self.sizes[FC.TARGET]

    @property
    def n_tpoints(self):
        """
        The number of points per target

        Returns
        -------
        int:
            The number of points per target

        """
        return self.sizes[FC.TPOINT]

    def tpoint_mean(self, variable):
        """
        Take the mean over target points

        Parameters
        ----------
        variable: str
            The variable name

        Returns
        -------
        data: numpy.ndarray
            The reduced array, shape:
            (n_states, n_targets, ...)

        """
        return np.einsum("stp...,p->st...", self[variable], self[FC.TWEIGHTS])

    @classmethod
    def from_points(
        cls,
        points,
        data={},
        dims={},
        name="tdata",
        **kwargs,
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
        kwargs: dict, optional
            Additional parameters for the constructor

        Returns
        -------
        pdata: Data
            The data object

        """
        if len(points.shape) != 3 or points.shape[2] != 3:
            raise ValueError(
                f"Expecting points shape (n_states, n_points, 3), got {points.shape}"
            )
        data[FC.TARGETS] = points[:, :, None, :]
        dims[FC.TARGETS] = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
        data[FC.TWEIGHTS] = np.array([1], dtype=FC.DTYPE)
        dims[FC.TWEIGHTS] = (FC.TPOINT,)
        return cls(data, dims, [FC.STATE, FC.TARGET], name=name, **kwargs)

    @classmethod
    def from_tpoints(
        cls,
        tpoints,
        tweights,
        data={},
        dims={},
        name="tdata",
        **kwargs,
    ):
        """
        Create from points at targets

        Parameters
        ----------
        tpoints: np.ndarray
            The points at targets, shape:
            (n_states, n_targets, n_tpoints, 3)
        tweights: np.ndarray, optional
            The target point weights, shape:
            (n_tpoints,)
        data: dict
            The initial data to be stored
        dims: dict
            The dimensions tuples, same or subset
            of data keys
        name: str
            The data container name
        kwargs: dict, optional
            Additional parameters for the constructor

        Returns
        -------
        pdata: Data
            The data object

        """
        if len(tpoints.shape) != 4 or tpoints.shape[3] != 3:
            raise ValueError(
                f"Expecting tpoints shape (n_states, n_targets, n_tpoints, 3), got {tpoints.shape}"
            )
        data[FC.TARGETS] = tpoints
        dims[FC.TARGETS] = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
        data[FC.TWEIGHTS] = tweights
        dims[FC.TWEIGHTS] = (FC.TPOINT,)
        return cls(data, dims, [FC.STATE], name=name, **kwargs)
