import numpy as np
from xarray import Dataset

from foxes.utils import Dict
from foxes.utils.memory_utils import (
    deep_split_by_nbytes,
    deep_update,
    get_object_nbytes,
)
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class Data(Dict[str, np.ndarray]):
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
    chunki_states: int, optional
        The index of the states chunk
    chunki_points: int, optional
        The index of the points chunk
    extra_data: dict, optional
        Additional data that is not dimensioned

    :group: core

    """

    def __init__(
        self,
        data={},
        dims={},
        loop_dims=[FC.STATE],
        states_i0=None,
        chunki_states=None,
        chunki_points=None,
        n_chunks_states=None,
        n_chunks_points=None,
        extra_data={},
        raw=False,
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
        chunki_states: int, optional
            The index of the states chunk
        chunki_points: int, optional
            The index of the points chunk
        n_chunks_states: int, optional
            The number of states chunks
        n_chunks_points: int, optional
            The number of points chunks
        extra_data: dict, optional
            Additional data that is not dimensioned
        raw: bool
            If True, skip the data checks and auto update
        name: str
            The data container name

        """
        super().__init__(_name=name)

        self.update(data)
        self.dims = dims
        self.loop_dims = loop_dims
        self.extra_data = extra_data

        self.__states_i0 = states_i0
        self.__chunki_states = chunki_states
        self.__chunki_points = chunki_points
        self.__n_chunks_states = n_chunks_states
        self.__n_chunks_points = n_chunks_points

        self.sizes = {}
        if not raw:
            for v, d in data.items():
                self._run_entry_checks(v, d, dims[v])
            self._auto_update()

    def to_dataset(self):
        """
        Convert to xarray.Dataset

        Returns
        -------
        ds: xarray.Dataset
            The dataset

        """
        return Dataset(
            data_vars={
                v: (self.dims[v], self[v]) for v in self.keys() if v not in self.sizes
            },
            coords={c: self[c] for c in self.sizes.keys()},
            attrs=self.extra_data,
        )

    def __str__(self):
        def _fmt_size(nbytes):
            if nbytes >= 1024 * 1024:
                return f"{nbytes / (1024 * 1024):.0f}MB"
            return f"{nbytes / 1024:.0f}kB"

        def _summary(value, level=0):
            if isinstance(value, np.ndarray):
                return f"array {value.dtype} {tuple(value.shape)}"
            if isinstance(value, dict):
                if level >= 1:
                    return f"dict(len={len(value)})"
                keys = sorted(value.keys(), key=lambda x: str(x))
                items = []
                max_items = 5
                for k in keys[:max_items]:
                    items.append(f"{k}: {_summary(value[k], level=level + 1)}")
                if len(keys) > max_items:
                    items.append("...")
                return f"dict(len={len(value)}){{{', '.join(items)}}}"
            if isinstance(value, (list, tuple, set)):
                return f"{type(value).__name__}(len={len(value)})"
            if isinstance(value, str):
                return f"str(len={len(value)})"
            if isinstance(value, np.generic):
                return f"{type(value).__name__}({value.item()})"
            return type(value).__name__

        def _dims_text(dims):
            if dims is None:
                return ""
            return f"({', '.join(dims)})"

        def _edge_preview(value):
            if isinstance(value, np.ndarray):
                if value.size == 0:
                    return "[]"
                flat = value.reshape(-1)
                first = flat[0].item() if isinstance(flat[0], np.generic) else flat[0]
                if value.size == 1:
                    return f"[{first}]"
                last = flat[-1].item() if isinstance(flat[-1], np.generic) else flat[-1]
                return f"[{first}...{last}]"

            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return "[]"
                if len(value) == 1:
                    return f"[{value[0]}]"
                return f"[{value[0]}...{value[-1]}]"

            return ""

        def _iter_extra_entries(data, level=0):
            for k in sorted(data.keys(), key=lambda x: str(x)):
                key = str(k)
                value = data[k]
                yield level, key, value
                if isinstance(value, dict):
                    yield from _iter_extra_entries(value, level=level + 1)

        total_nbytes = (
            get_object_nbytes(self)
            + get_object_nbytes(self.dims)
            + get_object_nbytes(self.sizes)
            + get_object_nbytes(self.extra_data)
        )

        lines = [
            f"<foxes.core.{type(self).__name__}> {self.name} {_fmt_size(total_nbytes)}",
        ]

        if self.sizes:
            dim_text = ", ".join([f"{k}: {v}" for k, v in sorted(self.sizes.items())])
            lines.append(f"Dimensions: ({dim_text})")
        else:
            lines.append("Dimensions: ()")

        coord_keys = sorted([k for k in self.keys() if k in self.sizes])
        if coord_keys:
            lines.append("Coordinates:")
            for k in coord_keys:
                vsize = _fmt_size(get_object_nbytes(self[k]))
                vedges = _edge_preview(self[k])
                vedges = f" {vedges}" if vedges else ""
                lines.append(f"  * {k:<12} {_summary(self[k])}{vedges} {vsize}")

        data_keys = sorted([k for k in self.keys() if k not in self.sizes])
        if data_keys:
            lines.append("Data variables:")
            for k in data_keys:
                vsize = _fmt_size(get_object_nbytes(self[k]))
                vedges = _edge_preview(self[k])
                vedges = f" {vedges}" if vedges else ""
                lines.append(
                    f"    {k:<12} {_dims_text(self.dims.get(k, None)):<16} {_summary(self[k])}{vedges} {vsize}"
                )

        lines.append("Extra data:")
        if self.extra_data:
            for level, k, v in _iter_extra_entries(self.extra_data):
                vedges = _edge_preview(v)
                vedges = f" {vedges}" if vedges else ""
                vsize = _fmt_size(get_object_nbytes(v))
                s = _summary(v, level=1) if isinstance(v, dict) else _summary(v)
                indent = "    " + "  " * level
                lines.append(f"{indent}{k}  {s}{vedges}  {vsize}")
        else:
            lines.append("    (none)")

        return "\n".join(lines)

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
    def chunki_states(self):
        """
        The index of the states chunk

        Returns
        -------
        int:
            The index of the states chunk

        """
        return self.__chunki_states

    @property
    def chunki_points(self):
        """
        The index of the points chunk

        Returns
        -------
        int:
            The index of the points chunk

        """
        return self.__chunki_points

    @property
    def n_chunks_states(self):
        """
        The number of states chunks

        Returns
        -------
        int:
            The number of states chunks

        """
        return self.__n_chunks_states

    @property
    def n_chunks_points(self):
        """
        The number of points chunks

        Returns
        -------
        int:
            The number of points chunks

        """
        return self.__n_chunks_points

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
        if counter:
            if self.__states_i0 is None:
                raise KeyError(f"Data '{self.name}': states_i0 requested but not set")
            return self.__states_i0
        elif FC.STATE not in self:
            return None
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
                for li, ld in enumerate(self.loop_dims):
                    if data.shape[li] == 1 and (len(dims) < li + 1 or dims[li] != ld):
                        self[name] = np.squeeze(data, axis=li)
            for ci, c in enumerate(dims):
                if c not in self.sizes or self.sizes[c] == 1:
                    self.sizes[c] = self[name].shape[ci]
                elif c != FC.TARGET and self[name].shape[ci] == 1:
                    pass
                elif (
                    self.sizes[c] != self[name].shape[ci] and self[name].shape[ci] != 1
                ):
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

    def get_slice(self, variables, s, dim_map={}, name=None, force=False):
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
        force: bool, optional
            Force the slice operation even if checks fail

        Returns
        -------
        data: Data
            The new data object, containing slices

        """
        if not isinstance(variables, (list, tuple, np.ndarray)):
            variables = [variables]
            s = [s]
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
                if force:
                    states_i0 = sts[0]
                else:
                    raise ValueError(
                        f"Cannot determine states_i0 for states slices {a}, leading to selection {list(sts)}"
                    )
        else:
            states_i0 = None

        cls = type(self)
        if issubclass(cls, Data):
            return cls(
                data,
                dims,
                name=name,
                states_i0=states_i0,
                chunki_states=self.chunki_states,
                chunki_points=self.chunki_points,
                n_chunks_states=self.n_chunks_states,
                n_chunks_points=self.n_chunks_points,
            )
        else:
            return cls(
                data,
                dims,
                loop_dims=self.loop_dims,
                name=name,
                states_i0=states_i0,
                chunki_states=self.chunki_states,
                chunki_points=self.chunki_points,
                n_chunks_states=self.n_chunks_states,
                n_chunks_points=self.n_chunks_points,
            )

    def pop_shared(self, min_shared_array_bytes=65536):
        """
        Pop the shared data, i.e. the data that is independent of the loop variables.

        Parameters
        ----------
        min_shared_array_bytes: int
            Minimum array size in bytes for moving loop-independent arrays into
            the shared data container. Smaller arrays stay in the current data
            object. The threshold is also applied recursively to ``extra_data``
            values.

        Returns
        -------
        shared: Data
            The shared data

        """
        data = {}
        dims = {}
        vrs = set(self.keys())
        for v in vrs:
            d = self.dims[v]
            if (
                d is not None
                and all([dd not in self.loop_dims for dd in d])
                and self[v].nbytes > min_shared_array_bytes
            ):
                data[v] = self.pop(v)
                dims[v] = self.dims.pop(v)

        # split extra data by size:
        self.extra_data, extra_data = deep_split_by_nbytes(
            self.extra_data,
            max_nbytes=min_shared_array_bytes + 1,
            fill_None=True,
        )

        shared = type(self)(
            data,
            dims,
            extra_data=extra_data,
            raw=True,
            name=self.name + "_shared",
        )

        return shared

    def recombine_with_shared(self, shared):
        """
        Recombine with shared data, i.e. add the shared data entries to the current data.

        Parameters
        ----------
        shared: Data
            The shared data

        """

        for v in shared.keys():
            if v in self:
                raise KeyError(
                    f"Cannot recombine with shared data, entry '{v}' already exists in data"
                )
            self[v] = shared[v]
            self.dims[v] = shared.dims[v]

        self.extra_data = deep_update(self.extra_data, shared.extra_data)

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
                n_states = d.shape[0]
                s = np.s_[:] if s_states is None else s_states
                data[v] = d.to_numpy()[s].copy() if copy else d.to_numpy()[s]
                dims[v] = d.dims
                if v == FV.WEIGHT and d.dims == (FC.STATE,):
                    data[v] = data[v][:, None]
                    dims[v] = (FC.STATE, FC.TURBINE)
            else:
                data[v] = d.to_numpy().copy() if copy else d.to_numpy()
                dims[v] = d.dims

        if callback is not None:
            callback(data, dims)

        if FC.STATE not in data and s_states is not None and n_states is not None:
            data[FC.STATE] = np.arange(n_states)[s_states]
            dims[FC.STATE] = (FC.STATE,)

        return cls(*args, data=data, dims=dims, **kwargs)

    @classmethod
    def from_data(cls, base_data, *args, callback=None, **kwargs):
        """
        Create Data object from another data object.

        Parameters
        ----------
        base_data: Data
            The source data
        args: tuple, optional
            Additional parameters for the constructor
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
        out = cls(
            *args,
            states_i0=base_data.states_i0,
            chunki_states=base_data.chunki_states,
            chunki_points=base_data.chunki_points,
            n_chunks_states=base_data.n_chunks_states,
            n_chunks_points=base_data.n_chunks_points,
            **kwargs,
        )

        for v in base_data.loop_dims:
            out[v] = base_data[v]
            out.dims[v] = base_data.dims[v]
            out.sizes[v] = base_data.sizes[v]

        if callback is not None:
            callback(out, out.dims)

        return out


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
        super().__init__(*args, loop_dims=[FC.STATE], name=name, **kwargs)

    def _run_entry_checks(self, name, data, dims):
        """Run entry checks on new data"""
        super()._run_entry_checks(name, data, dims)
        data = self[name]
        dims = self.dims[name]
        if name not in self.sizes and name not in [FC.TNAME, FV.WEIGHT]:
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
    def from_sizes(cls, n_states, n_turbines, *args, callback=None, **kwargs):
        """
        Create Data object from model data

        Parameters
        ----------
        n_states: int
            The number of states
        n_turbines: int
            The number of turbines
        args: tuple, optional
            Additional parameters for the constructor
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
        data = cls(*args, **kwargs)
        data.sizes[FC.STATE] = n_states
        data.sizes[FC.TURBINE] = n_turbines

        if callback is not None:
            callback(data, data.dims)

        return data

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
                if FC.TURBINE not in data:
                    data[FC.TURBINE] = mdata[FC.TURBINE]
                    dims[FC.TURBINE] = mdata.dims[FC.TURBINE]
                if callback is not None:
                    callback(data, dims)

            return super().from_dataset(
                ds,
                *args,
                callback=cb,
                chunki_states=mdata.chunki_states,
                chunki_points=mdata.chunki_points,
                n_chunks_states=mdata.n_chunks_states,
                n_chunks_points=mdata.n_chunks_points,
                **kwargs,
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
        super().__init__(*args, loop_dims=[FC.STATE, FC.TARGET], name=name, **kwargs)

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

    def get_targets_subset(self, sel_targets):
        """
        Get a subset of targets

        Parameters
        ----------
        sel_targets: array_like of int
            The target indices to select

        Returns
        -------
        tdata: TData
            The new TData object with the selected targets

        """
        data = {}
        dims = {}
        for v in self.keys():
            if v in self.dims and FC.TARGET in self.dims[v]:
                if len(self.dims[v]) >= 2 and self.dims[v][1] == FC.TARGET:
                    if self.n_targets > 1 and self[v].shape[1] > 1:
                        data[v] = self[v][:, sel_targets, ...]
                    else:
                        data[v] = self[v]
                elif len(self.dims[v]) >= 1 and self.dims[v][0] == FC.TARGET:
                    if self.n_targets > 1 and self[v].shape[0] > 1:
                        data[v] = self[v][sel_targets, ...]
                    else:
                        data[v] = self[v]
                else:
                    raise ValueError(
                        f"TData '{self.name}': Cannot subset variable '{v}' with dims {self.dims[v]} for target selection, expecting '{FC.TARGET}' in dims at positions 0 or 1"
                    )
                dims[v] = self.dims[v]
            else:
                data[v] = self[v]
                dims[v] = self.dims[v]

        try:
            states_i0 = self.states_i0(counter=True)
        except KeyError:
            states_i0 = None

        return self.__class__(
            data=data,
            dims=dims,
            name=f"{self.name}_subset",
            states_i0=states_i0,
            chunki_states=self.chunki_states,
            chunki_points=self.chunki_points,
            n_chunks_states=self.n_chunks_states,
            n_chunks_points=self.n_chunks_points,
        )

    @classmethod
    def from_points(
        cls,
        points,
        data=None,
        dims=None,
        variables=None,
        mdata=None,
        name="tdata",
        **kwargs,
    ):
        """
        Create from points

        Parameters
        ----------
        points: np.ndarray
            The points, shape: (n_states, n_points, 3)
        data: dict, optional
            The initial data to be stored
        dims: dict, optional
            The dimensions tuples, same or subset
            of data keys
        variables: list of str
            Add default empty variables with NaN values
            and shape (n_states, n_targets, n_tpoints)
        mdata: MData, optional
            The model data
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
        data = {} if data is None else data
        dims = {} if dims is None else dims
        data[FC.TARGETS] = points[:, :, None, :]
        dims[FC.TARGETS] = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
        data[FC.TWEIGHTS] = np.array([1], dtype=config.dtype_double)
        dims[FC.TWEIGHTS] = (FC.TPOINT,)
        if variables is not None:
            for v in variables:
                data[v] = np.full_like(points[:, :, None, 0], np.nan)
                dims[v] = (FC.STATE, FC.TARGET, FC.TPOINT)

        if mdata is not None:
            kwargs["chunki_states"] = mdata.chunki_states
            kwargs["chunki_points"] = mdata.chunki_points
            kwargs["n_chunks_states"] = mdata.n_chunks_states
            kwargs["n_chunks_points"] = mdata.n_chunks_points

        return cls(data=data, dims=dims, name=name, **kwargs)

    @classmethod
    def from_tpoints(
        cls,
        tpoints,
        tweights,
        data=None,
        dims=None,
        variables=None,
        mdata=None,
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
        data: dict, optional
            The initial data to be stored
        dims: dict, optional
            The dimensions tuples, same or subset
            of data keys
        variables: list of str
            Add default empty variables with NaN values
            and shape (n_states, n_targets, n_tpoints)
        mdata: MData, optional
            The model data
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
        data = {} if data is None else data
        dims = {} if dims is None else dims
        data[FC.TARGETS] = tpoints
        dims[FC.TARGETS] = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
        data[FC.TWEIGHTS] = tweights
        dims[FC.TWEIGHTS] = (FC.TPOINT,)
        if variables is not None:
            for v in variables:
                data[v] = np.full_like(tpoints[..., 0], np.nan)
                dims[v] = (FC.STATE, FC.TARGET, FC.TPOINT)

        if mdata is not None:
            kwargs["chunki_states"] = mdata.chunki_states
            kwargs["chunki_points"] = mdata.chunki_points
            kwargs["n_chunks_states"] = mdata.n_chunks_states
            kwargs["n_chunks_points"] = mdata.n_chunks_points

        return cls(data=data, dims=dims, name=name, **kwargs)

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
            kwargs["chunki_states"] = mdata.chunki_states
            kwargs["chunki_points"] = mdata.chunki_points
            kwargs["n_chunks_states"] = mdata.n_chunks_states
            kwargs["n_chunks_points"] = mdata.n_chunks_points

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
                                f"Expecting coordinates '{(FC.STATE, FC.TARGET, FC.TPOINT)}' at positions 0-2 for data variable '{v}', got {dims[v]}"
                            )
                        else:
                            data[v] = d[:, s_targets]
                if cb0 is not None:
                    cb0(data, dims)

            cb1 = cb_targets

        return super().from_dataset(ds, *args, callback=cb1, **kwargs)
