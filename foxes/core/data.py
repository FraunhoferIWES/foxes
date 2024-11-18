import numpy as np

from foxes.utils import Dict
from foxes.config import config
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

    def __init__(
        self,
        data,
        dims,
        loop_dims,
        states_i0=None,
        name="data",
    ):
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
        states_i0: int, optional
            The index of the first state
        name: str
            The data container name

        """
        super().__init__(name=name)

        self.update(data)
        self.dims = dims
        self.loop_dims = loop_dims

        self.__states_i0 = states_i0

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

    def states_i0(self, counter=False):
        """
        Get the state counter for first state in chunk

        Parameters
        ----------
        counter: bool
            Return the state counter instead of the index

        Returns
        -------
        int:
            The state counter for first state in chunk
            or the corresponding index

        """
        if FC.STATE not in self:
            return None
        elif counter:
            if self.__states_i0 is None:
                raise KeyError(f"Data '{self.name}': states_i0 requested but not set")
            return self.__states_i0
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

        allc = set()
        for dms in self.dims.values():
            if dms is not None:
                allc.update(dms)
        allc = allc.difference(set(data.keys()))
        for c in allc.intersection(self.sizes.keys()):
            data[c] = np.arange(self.sizes[c])
            dims[c] = (c,)

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

    def get_slice(self, variables, s, dim_map={}, name=None):
        """
        Get a slice of data.

        Parameters
        ----------
        variables: list of str
            The variable list that corresponds to s
        s: slice
            The slice
        dim_map: dict
            Mapping from original to new dimensions.
            If not found, same dimensions are assumed.
        name: str, optional
            The name of the data object

        Returns
        -------
        data: Data
            The new data object, containing slices

        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            variables = [variables]
        if not isinstance(s, (list, tuple, np.ndarray)):
            s = [s]

        data = {}
        dims = {}
        for v in self.keys():
            d = self.dims[v]
            if d is not None:
                hs = tuple(
                    [s[variables.index(w)] if w in variables else np.s_[:] for w in d]
                )
                data[v] = self[v][hs]
                dims[v] = (
                    tuple([dim_map.get(dd, dd) for dd in d]) if len(dim_map) else d
                )
        if name is None:
            name = self.name
        if FC.STATE in variables and self.__states_i0 is not None:
            i0 = self.states_i0(counter=True)
            a = s[variables.index(FC.STATE)]
            sts = np.arange(i0, i0 + self.n_states)[a]
            if len(sts) == 1:
                states_i0 = sts[0]
            elif np.all(sts == np.arange(sts[0], sts[0] + len(sts))):
                states_i0 = sts[0]
            else:
                raise ValueError(
                    f"Cannot determine states_i0 for states slices {a}, leading to selection {list(sts)}"
                )
        else:
            states_i0 = None
        return type(self)(
            data, dims, loop_dims=self.loop_dims, name=name, states_i0=states_i0
        )

    @classmethod
    def from_dataset(cls, ds, *args, callback=None, s_states=None, copy=True, **kwargs):
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset
        args: tuple, optional
            Additional parameters for the constructor
        callback: Function, optional
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        s_states: slice, optional
            Slice object for states
        copy: bool
            Flag for copying data
        kwargs: dict, optional
            Additional parameters for the constructor

        Returns
        -------
        data: Data
            The data object

        """
        data = {}
        dims = {}

        for c, d in ds.coords.items():
            if c == FC.STATE:
                s = np.s_[:] if s_states is None else s_states
                data[c] = d.to_numpy()[s].copy() if copy else d.to_numpy()[s]
            else:
                data[c] = d.to_numpy().copy() if copy else d.to_numpy()
            dims[c] = d.dims

        n_states = None
        for v, d in ds.data_vars.items():
            if FC.STATE in d.dims:
                if d.dims[0] != FC.STATE:
                    raise ValueError(
                        f"Expecting coordinate '{FC.STATE}' at position 0 for data variable '{v}', got {d.dims}"
                    )
                n_states = len(d.to_numpy())
                s = np.s_[:] if s_states is None else s_states
                data[v] = d.to_numpy()[s].copy() if copy else d.to_numpy()[s]
            else:
                data[v] = d.to_numpy().copy() if copy else d.to_numpy()
            dims[v] = d.dims

        if callback is not None:
            callback(data, dims)

        if FC.STATE not in data and s_states is not None and n_states is not None:
            data[FC.STATE] = np.arange(n_states)[s_states]
            dims[FC.STATE] = (FC.STATE,)

        return cls(*args, data=data, dims=dims, **kwargs)


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

    @classmethod
    def from_dataset(cls, ds, *args, mdata=None, callback=None, **kwargs):
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset
        args: tuple, optional
            Additional parameters for the constructor
        mdata: MData, optional
            The mdata object
        callback: Function, optional
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        kwargs: dict, optional
            Additional parameters for the constructor

        Returns
        -------
        data: Data
            The data object

        """
        if mdata is None:
            return super().from_dataset(ds, *args, callback=callback, **kwargs)
        else:

            def cb(data, dims):
                if FC.STATE not in data:
                    data[FC.STATE] = mdata[FC.STATE]
                    dims[FC.STATE] = mdata.dims[FC.STATE]
                    data[FV.WEIGHT] = mdata[FV.WEIGHT]
                    dims[FV.WEIGHT] = mdata.dims[FV.WEIGHT]
                if callback is not None:
                    callback(data, dims)

            return super().from_dataset(ds, *args, callback=cb, **kwargs)


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

    def targets_i0(self):
        """
        Get the target counter for first target in chunk

        Returns
        -------
        int:
            The target index for first target in chunk

        """
        if FC.TARGET not in self:
            return None
        else:
            return self[FC.TARGET][0]

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
        data[FC.TWEIGHTS] = np.array([1], dtype=config.dtype_double)
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

    @classmethod
    def from_dataset(
        cls,
        ds,
        *args,
        s_targets=None,
        mdata=None,
        callback=None,
        **kwargs,
    ):
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset
        args: tuple, optional
            Additional parameters for the constructor
        s_targets: slice, optional
            Slice object for targets
        mdata: MData, optional
            The mdata object
        callback: Function, optional
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        kwargs: dict, optional
            Additional parameters for the constructor

        Returns
        -------
        data: Data
            The data object

        """
        if mdata is None:
            cb0 = callback
        else:

            def cb_mdata(data, dims):
                if FC.STATE not in data:
                    data[FC.STATE] = mdata[FC.STATE]
                    dims[FC.STATE] = mdata.dims[FC.STATE]
                if callback is not None:
                    callback(data, dims)

            cb0 = cb_mdata

        if s_targets is None:
            cb1 = cb0
        else:

            def cb_targets(data, dims):
                if FC.TARGET not in data:
                    data[FC.TARGET] = np.arange(ds.sizes[FC.TARGET])
                    dims[FC.TARGET] = (FC.TARGET,)
                for v, d in data.items():
                    if FC.TARGET in dims[v]:
                        if dims[v] == (FC.TARGET,):
                            data[v] = d[s_targets].copy()
                        elif len(dims[v]) < 3 or dims[v][:3] != (
                            FC.STATE,
                            FC.TARGET,
                            FC.TPOINT,
                        ):
                            raise ValueError(
                                f"Expecting coordinates '{ (FC.STATE, FC.TARGET, FC.TPOINT)}' at positions 0-2 for data variable '{v}', got {dims[v]}"
                            )
                        else:
                            data[v] = d[:, s_targets]
                if cb0 is not None:
                    cb0(data, dims)

            cb1 = cb_targets

        return super().from_dataset(ds, *args, callback=cb1, **kwargs)
