from __future__ import annotations

import numpy as np
from xarray import Dataset
from typing import Any, Callable

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
    dims
        The dimensions tuples, same or subset
        of data keys
    loop_dims
        Loop dimensions used during xarray's `apply_ufunc` calculations
    sizes
        The dimension sizes
    chunki_states
        The index of the states chunk
    chunki_points
        The index of the points chunk
    extra_data
        Additional data that is not dimensioned


    """

    def __init__(
        self,
        data: dict[str, np.ndarray] | None = None,
        dims: dict[str, tuple[str, ...]] | None = None,
        loop_dims: list[str] | None = None,
        states_i0: int | None = None,
        chunki_states: int | None = None,
        chunki_points: int | None = None,
        n_chunks_states: int | None = None,
        n_chunks_points: int | None = None,
        extra_data: dict[str, Any] | None = None,
        raw: bool = False,
        name: str = "data",
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data
            The initial data to be stored.
        dims
            The dimension tuples, same as or a subset of the data keys.
        loop_dims
            The loop dimensions during xarray ``apply_ufunc`` calculations.
        states_i0
            The index of the first state.
        chunki_states
            The index of the states chunk.
        chunki_points
            The index of the points chunk.
        n_chunks_states
            The number of states chunks.
        n_chunks_points
            The number of points chunks.
        extra_data
            Additional data that is not dimensioned.
        raw
            If ``True``, skip the data checks and auto-update logic.
        name
            The data container name.

        """
        super().__init__(_name=name)

        data = {} if data is None else data
        dims = {} if dims is None else dims
        loop_dims = [FC.STATE] if loop_dims is None else loop_dims
        extra_data = {} if extra_data is None else extra_data

        self.update(data)
        self.dims = dims
        self.loop_dims = loop_dims
        self.extra_data = extra_data

        self.__states_i0 = states_i0
        self.__chunki_states = chunki_states
        self.__chunki_points = chunki_points
        self.__n_chunks_states = n_chunks_states
        self.__n_chunks_points = n_chunks_points

        self.sizes: dict[str, int] = {}
        if not raw:
            for v, d in data.items():
                self._run_entry_checks(v, d, dims[v])
            self._auto_update()

    def to_dataset(self) -> Dataset:
        """
        Convert to xarray.Dataset

        Returns
        -------
        ds
            The dataset

        """
        return Dataset(
            data_vars={
                v: (self.dims[v], self[v]) for v in self.keys() if v not in self.sizes
            },
            coords={c: self[c] for c in self.sizes.keys()},
            attrs=self.extra_data,
        )

    def __str__(self) -> str:
        def _fmt_size(nbytes: int) -> str:
            units = ("B", "KB", "MB", "GB", "TB", "PB")
            size = float(nbytes)
            ui = 0
            while size >= 1024.0 and ui < len(units) - 1:
                size /= 1024.0
                ui += 1
            if ui == 0:
                return f"{int(size)}{units[ui]}"
            return f"{size:.2f}{units[ui]}"

        def _summary(value: Any, level: int = 0) -> str:
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

        def _dims_text(dims: tuple[str, ...] | None) -> str:
            if dims is None:
                return ""
            return f"({', '.join(dims)})"

        def _edge_preview(value: Any) -> str:
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

        def _iter_extra_entries(data: dict[str, Any], level: int = 0) -> Any:
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
    def n_states(self) -> int | None:
        """
        The number of states

        Returns
        -------
        int:
            The number of states

        """
        return self.sizes[FC.STATE] if FC.STATE in self.sizes else None

    @property
    def n_turbines(self) -> int | None:
        """
        The number of turbines

        Returns
        -------
        int:
            The number of turbines

        """
        return self.sizes[FC.TURBINE] if FC.TURBINE in self.sizes else None

    @property
    def chunki_states(self) -> int | None:
        """
        The index of the states chunk

        Returns
        -------
        int:
            The index of the states chunk

        """
        return self.__chunki_states

    @property
    def chunki_points(self) -> int | None:
        """
        The index of the points chunk

        Returns
        -------
        int:
            The index of the points chunk

        """
        return self.__chunki_points

    @property
    def n_chunks_states(self) -> int | None:
        """
        The number of states chunks

        Returns
        -------
        int:
            The number of states chunks

        """
        return self.__n_chunks_states

    @property
    def n_chunks_points(self) -> int | None:
        """
        The number of points chunks

        Returns
        -------
        int:
            The number of points chunks

        """
        return self.__n_chunks_points

    def states_i0(self, counter: bool = False) -> int | None:
        """
        Get the state counter for first state in chunk

        Parameters
        ----------
        counter
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

    def _auto_update(self) -> None:
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

        allc: set[str] = set()
        for dms in self.dims.values():
            if dms is not None:
                allc.update(dms)
        allc = allc.difference(set(data.keys()))
        for c in allc.intersection(self.sizes.keys()):
            data[c] = np.arange(self.sizes[c])
            dims[c] = (c,)

    def _run_entry_checks(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...] | None,
    ) -> None:
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

    def add(self, name: str, data: np.ndarray, dims: tuple[str, ...]) -> None:
        """
        Add data entry

        Parameters
        ----------
        name
            The data name
        data
            The data
        dims
            The dimensions

        """
        self[name] = data
        self.dims[name] = dims
        self._run_entry_checks(name, data, dims)
        self._auto_update()

    def get_slice(
        self,
        variables: Any,
        s: Any,
        dim_map: dict[str, str] | None = None,
        name: str | None = None,
        force: bool = False,
    ) -> Data:
        """
        Get a slice of data.

        Parameters
        ----------
        variables
            The variable list that corresponds to the selected slice.
        s
            The slice specification.
        dim_map
            Mapping from original to new dimensions. If not found, the same
            dimensions are assumed.
        name
            The name of the data object.
        force
            Force the slice operation even if checks fail.

        Returns
        -------
        data
            The new data object containing the slices.

        """
        if dim_map is None:
            dim_map = {}
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
            assert i0 is not None
            assert self.n_states is not None
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

    def pop_shared(self, min_shared_array_bytes: int = 65536) -> Data:
        """
        Pop the shared data, i.e. the data that is independent of the loop variables.

        Parameters
        ----------
        min_shared_array_bytes
            Minimum array size in bytes for moving loop-independent arrays into
            the shared data container. Smaller arrays stay in the current data
            object. The threshold is also applied recursively to ``extra_data``
            values.

        Returns
        -------
        shared
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

    def recombine_with_shared(self, shared: Data) -> None:
        """
        Recombine with shared data, i.e. add the shared data entries to the current data.

        Parameters
        ----------
        shared
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
    def from_dataset(
        cls,
        ds: Dataset,
        *args: Any,
        callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        s_states: Any = None,
        copy: bool = True,
        n_states: int | None = None,
        n_turbines: int | None = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds
            The dataset
        args
            Additional parameters for the constructor
        callback
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        s_states
            Optional slice object for states
        copy
            Flag for copying data
        n_states
            The number of states, if not found in the dataset
        n_turbines
            The number of turbines, if not found in the dataset
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        data
            The data object

        """
        data: dict[str, Any] = {}
        dims: dict[str, tuple[str, ...]] = {}
        if n_states == 0:
            raise ValueError("Cannot create Data object with n_states=0")

        for c, d in ds.coords.items():
            c_name = str(c)
            if c_name == FC.STATE:
                s = np.s_[:] if s_states is None else s_states
                data[c_name] = d.to_numpy()[s].copy() if copy else d.to_numpy()[s]
            else:
                data[c_name] = d.to_numpy().copy() if copy else d.to_numpy()
            dims[c_name] = tuple(str(dd) for dd in d.dims)

        for v, d in ds.data_vars.items():
            v_name = str(v)
            if FC.STATE in d.dims:
                if d.dims[0] != FC.STATE:
                    raise ValueError(
                        f"Expecting coordinate '{FC.STATE}' at position 0 for data variable '{v_name}', got {d.dims}"
                    )
                s = np.s_[:] if s_states is None else s_states
                data[v_name] = d.to_numpy()[s].copy() if copy else d.to_numpy()[s]
                dims[v_name] = tuple(str(dd) for dd in d.dims)
                if n_states is None or n_states == 1:
                    n_states = data[v_name].shape[0]
                elif data[v_name].shape[0] == 1:
                    pass
                else:
                    assert n_states == data[v_name].shape[0], (
                        f"Expecting {n_states} states, got {data[v_name].shape[0]} in data variable '{v_name}'"
                    )
                if v_name == FV.WEIGHT and d.dims == (FC.STATE,):
                    data[v_name] = data[v_name][:, None]
                    dims[v_name] = (FC.STATE, FC.TURBINE)
            else:
                data[v_name] = d.to_numpy().copy() if copy else d.to_numpy()
                dims[v_name] = tuple(str(dd) for dd in d.dims)

        if FC.TURBINE not in data and n_turbines is not None:
            data[FC.TURBINE] = np.arange(n_turbines)
            dims[FC.TURBINE] = (FC.TURBINE,)

        if callback is not None:
            callback(data, dims)

        if FC.STATE not in data and n_states is not None:
            data[FC.STATE] = np.arange(n_states)
            dims[FC.STATE] = (FC.STATE,)

        return cls(*args, data=data, dims=dims, **kwargs)  # type: ignore[misc]

    @classmethod
    def from_data(
        cls,
        base_data: Data,
        *args: Any,
        callback: Callable[[Data, dict[str, Any]], None] | None = None,
        states_i0: int | None = None,
        chunki_states: int | None = None,
        chunki_points: int | None = None,
        n_chunks_states: int | None = None,
        n_chunks_points: int | None = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create Data object from another data object.

        Parameters
        ----------
        base_data
            The source data
        args
            Additional parameters for the constructor
        callback
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        states_i0
            The index of the first state. If omitted, copied from base_data
        chunki_states
            The index of the states chunk. If omitted, copied from base_data
        chunki_points
            The index of the points chunk. If omitted, copied from base_data
        n_chunks_states
            The total number of states chunks. If omitted, copied from base_data
        n_chunks_points
            The total number of points chunks. If omitted, copied from base_data
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        data
            The data object

        """
        if states_i0 is None:
            try:
                states_i0 = base_data.states_i0(counter=True)
            except KeyError:
                states_i0 = None
        if chunki_states is None:
            chunki_states = base_data.chunki_states
        if chunki_points is None:
            chunki_points = base_data.chunki_points
        if n_chunks_states is None:
            n_chunks_states = base_data.n_chunks_states
        if n_chunks_points is None:
            n_chunks_points = base_data.n_chunks_points

        out = cls(
            *args,
            states_i0=states_i0,
            chunki_states=chunki_states,
            chunki_points=chunki_points,
            n_chunks_states=n_chunks_states,
            n_chunks_points=n_chunks_points,
            **kwargs,
        )  # type: ignore[misc]

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


    """

    def __init__(self, *args: Any, name: str = "mdata", **kwargs: Any) -> None:
        """
        Constructor

        Parameters
        ----------
        args
            Arguments for the base class
        name
            The data name
        kwargs
            Arguments for the base class

        """
        super().__init__(*args, name=name, **kwargs)  # type: ignore[misc]


class FData(Data):
    """
    Container for foxes farm data.

    Each farm data entry has (n_states, n_turbines) shape,
    except the dimensions.


    """

    def __init__(self, *args: Any, name: str = "fdata", **kwargs: Any) -> None:
        """
        Constructor

        Parameters
        ----------
        args
            Arguments for the base class
        name
            The data name
        kwargs
            Arguments for the base class

        """
        super().__init__(*args, loop_dims=[FC.STATE], name=name, **kwargs)  # type: ignore[misc]

    def _run_entry_checks(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...] | None,
    ) -> None:
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

    def _auto_update(self) -> None:
        """Checks and operations after data changes"""
        super()._auto_update()
        if len(self):
            for x in [FC.STATE, FC.TURBINE]:
                if x not in self.sizes:
                    raise KeyError(
                        f"FData '{self.name}': Missing '{x}' in sizes, got {sorted(list(self.sizes.keys()))}"
                    )

    @classmethod
    def from_sizes(
        cls,
        n_states: int,
        n_turbines: int,
        *args: Any,
        callback: Callable[[Data, dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create Data object from model data

        Parameters
        ----------
        n_states
            The number of states
        n_turbines
            The number of turbines
        args
            Additional parameters for the constructor
        callback
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        data
            The data object

        """
        data = cls(*args, **kwargs)
        data.sizes[FC.STATE] = n_states
        data.sizes[FC.TURBINE] = n_turbines

        if callback is not None:
            callback(data, data.dims)

        return data

    @classmethod
    def from_dataset(
        cls,
        ds: Dataset,
        *args: Any,
        mdata: MData | None = None,
        callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        n_states: int | None = None,
        n_turbines: int | None = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds
            The dataset
        args
            Additional parameters for the constructor
        mdata
            The mdata object
        callback
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        n_states
            The number of states, if not found in the dataset
        n_turbines
            The number of turbines, if not found in the dataset
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        data
            The data object

        """
        if mdata is None:
            return super().from_dataset(ds, *args, callback=callback, **kwargs)
        else:

            def cb(data: dict[str, Any], dims: dict[str, Any]) -> None:
                if FC.STATE not in data:
                    if FC.STATE in mdata:
                        data[FC.STATE] = mdata[FC.STATE]
                        dims[FC.STATE] = mdata.dims[FC.STATE]
                    else:
                        assert n_states is not None, (
                            "n_states must be provided if not found in mdata"
                        )
                        i0 = mdata.states_i0(counter=True)
                        assert i0 is not None
                        data[FC.STATE] = np.arange(i0, i0 + n_states)
                        dims[FC.STATE] = (FC.STATE,)
                if FC.TURBINE not in data:
                    if FC.TURBINE in mdata:
                        data[FC.TURBINE] = mdata[FC.TURBINE]
                        dims[FC.TURBINE] = mdata.dims[FC.TURBINE]
                    else:
                        assert n_turbines is not None, (
                            "n_turbines must be provided if not found in mdata"
                        )
                        data[FC.TURBINE] = np.arange(n_turbines)
                        dims[FC.TURBINE] = (FC.TURBINE,)
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


    """

    def __init__(self, *args: Any, name: str = "tdata", **kwargs: Any) -> None:
        """
        Constructor

        Parameters
        ----------
        args
            Arguments for the base class
        name
            The data name
        kwargs
            Arguments for the base class

        """
        super().__init__(*args, loop_dims=[FC.STATE, FC.TARGET], name=name, **kwargs)  # type: ignore[misc]

    def _run_entry_checks(
        self,
        name: str,
        data: np.ndarray,
        dims: tuple[str, ...] | None,
    ) -> None:
        """Run entry checks on new data"""
        super()._run_entry_checks(name, data, dims)
        data = self[name]
        dims = self.dims[name]
        n_states = self.n_states
        assert n_states is not None

        if name == FC.TARGETS:
            dms: tuple[str, ...] = (FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)
            shp: tuple[int, ...] = (n_states, self.n_targets, self.n_tpoints, 3)
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
            shp = (n_states, self.n_targets, self.n_tpoints)
            if len(data.shape) < 3:
                raise ValueError(
                    f"TData '{self.name}': Invalid shape for '{name}', expecting {shp}, got {data.shape}"
                )
            if len(dims) < 3 or dims[:3] != dms:
                raise ValueError(
                    f"TData '{self.name}': Invalid dims for '{name}', expecting {dms}, got {dims}"
                )

    def _auto_update(self) -> None:
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
    def n_targets(self) -> int:
        """
        The number of targets

        Returns
        -------
        n_targets
            The number of targets

        """
        return self.sizes[FC.TARGET]

    @property
    def n_tpoints(self) -> int:
        """
        The number of points per target

        Returns
        -------
        n_tpoints
            The number of points per target

        """
        return self.sizes[FC.TPOINT]

    def tpoint_mean(self, variable: str) -> np.ndarray:
        """
        Take the mean over target points

        Parameters
        ----------
        variable
            The variable name

        Returns
        -------
        data
            The reduced array, shape:
            (n_states, n_targets, ...)

        """
        return np.einsum("stp...,p->st...", self[variable], self[FC.TWEIGHTS])

    def targets_i0(self) -> int | None:
        """
        Get the target counter for first target in chunk

        Returns
        -------
        targets_i0
            The target index for first target in chunk

        """
        if FC.TARGET not in self:
            return None
        else:
            return self[FC.TARGET][0]

    def get_targets_subset(self, sel_targets: Any) -> TData:
        """
        Get a subset of targets

        Parameters
        ----------
        sel_targets
            The target indices to select

        Returns
        -------
        tdata
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
        points: np.ndarray,
        data: dict[str, Any] | None = None,
        dims: dict[str, tuple[str, ...]] | None = None,
        variables: list[str] | None = None,
        mdata: MData | None = None,
        name: str = "tdata",
        **kwargs: Any,
    ) -> TData:
        """
        Create from points

        Parameters
        ----------
        points
            The points, shape: (n_states, n_points, 3)
        data
            The initial data to be stored
        dims
            The dimensions tuples, same or subset
            of data keys
        variables
            Add default empty variables with NaN values
            and shape (n_states, n_targets, n_tpoints)
        mdata
            The model data
        name
            The data container name
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        pdata
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
        tpoints: np.ndarray,
        tweights: np.ndarray,
        data: dict[str, Any] | None = None,
        dims: dict[str, tuple[str, ...]] | None = None,
        variables: list[str] | None = None,
        mdata: MData | None = None,
        name: str = "tdata",
        **kwargs: Any,
    ) -> TData:
        """
        Create from points at targets

        Parameters
        ----------
        tpoints
            The points at targets, shape:
            (n_states, n_targets, n_tpoints, 3)
        tweights
            The target point weights, shape:
            (n_tpoints,)
        data
            The initial data to be stored
        dims
            The dimensions tuples, same or subset
            of data keys
        variables
            Add default empty variables with NaN values
            and shape (n_states, n_targets, n_tpoints)
        mdata
            The model data
        name
            The data container name
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        pdata
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
        ds: Dataset,
        *args: Any,
        s_targets: Any = None,
        mdata: MData | None = None,
        callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create Data object from a dataset

        Parameters
        ----------
        ds
            The dataset
        args
            Additional parameters for the constructor
        s_targets
            Optional slice object for targets
        mdata
            The mdata object
        callback
            Function f(data, dims) that manipulates
            the data and dims dicts before construction
        kwargs
            Additional parameters for the constructor

        Returns
        -------
        data
            The data object

        """
        if mdata is None:
            cb0 = callback
        else:

            def cb_mdata(data: dict[str, Any], dims: dict[str, Any]) -> None:
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

            def cb_targets(data: dict[str, Any], dims: dict[str, Any]) -> None:
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
