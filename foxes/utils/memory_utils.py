import numpy as np
import sys
from collections.abc import Collection, Mapping, Sequence, Set as AbstractSet
from typing import Any

from .load import import_module


def get_object_nbytes(
    value: Any,
    recursive: bool = True,
    seen: set[int] | None = None,
    allow_shallow_fallback: bool = True,
) -> int:
    """Estimate payload bytes of nested extra_data values.

    The estimate is recursive for generic container types and focuses on
    payload-like elements (numpy arrays, bytes-like objects, nested
    containers), while ignoring scalar Python object overhead.

    Parameters
    ----------
    value
        The value to estimate the payload bytes for.
    recursive
        Whether to recursively estimate the payload bytes for nested containers.
    seen
        A set of object IDs that have already been seen to avoid double counting.
    allow_shallow_fallback
        If True, use ``sys.getsizeof`` as a shallow last-resort estimate for
        unknown object types. If False, return 0 for unknown object types to
        preserve payload-only semantics.


    """
    if seen is None:
        seen = set()

    vid = id(value)
    if vid in seen:
        return 0
    seen.add(vid)

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, np.generic):
        return int(value.dtype.itemsize)
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return len(value)
    if isinstance(value, Mapping):
        if not recursive:
            if not allow_shallow_fallback:
                return 0
            try:
                return int(sys.getsizeof(value))
            except (TypeError, ValueError):
                return 0
        return sum(
            get_object_nbytes(
                k,
                recursive=recursive,
                seen=seen,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            + get_object_nbytes(
                v,
                recursive=recursive,
                seen=seen,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            for k, v in value.items()
        )
    if isinstance(value, AbstractSet):
        if not recursive:
            if not allow_shallow_fallback:
                return 0
            try:
                return int(sys.getsizeof(value))
            except (TypeError, ValueError):
                return 0
        return sum(
            get_object_nbytes(
                v,
                recursive=recursive,
                seen=seen,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            for v in value
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, memoryview, str)
    ):
        if not recursive:
            if not allow_shallow_fallback:
                return 0
            try:
                return int(sys.getsizeof(value))
            except (TypeError, ValueError):
                return 0
        return sum(
            get_object_nbytes(
                v,
                recursive=recursive,
                seen=seen,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            for v in value
        )
    if isinstance(value, Collection) and not isinstance(
        value, (bytes, bytearray, memoryview, str)
    ):
        if not recursive:
            if not allow_shallow_fallback:
                return 0
            try:
                return int(sys.getsizeof(value))
            except (TypeError, ValueError):
                return 0
        return sum(
            get_object_nbytes(
                v,
                recursive=recursive,
                seen=seen,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            for v in value
        )

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, (int, np.integer)):
        return int(nbytes)

    if not allow_shallow_fallback:
        return 0

    try:
        return int(sys.getsizeof(value))
    except (TypeError, ValueError):
        return 0


def print_mem(
    obj: Any,
    min_csize: int = 0,
    max_csize: int | None = None,
    pre_str: str = "OBJECT SIZE",
) -> None:
    """
    Prints the memory consumption of a model and its components

    Parmeters
    ---------
    obj
        The object to be analyzed
    min_csize
        The minimal size of a component for being shown
    max_csize
        The maximal allowed size of a component
    pre_str
        String to be printed before


    """
    objsize = import_module("objsize")
    n = obj.name if hasattr(obj, "name") else ""
    print(pre_str, type(obj).__name__, n, objsize.get_deep_size(obj))
    for k in dir(obj):
        o = None
        try:
            if (
                hasattr(obj, k)
                and not callable(getattr(obj, k))
                and (len(k) < 3 or k[:2] != "__")
            ):
                o = getattr(obj, k)
        except ValueError:
            pass

        if o is not None:
            s = objsize.get_deep_size(getattr(obj, k))
            if s >= min_csize:
                print("   ", k, s)
                if max_csize is not None and s > max_csize:
                    raise ValueError(f"Component {k} exceeds maximal size {max_csize}")


def deep_split(condition: Any, data: Any, fill_None: bool = True) -> tuple[Any, Any]:
    """
    Recursively split data into two parts based on a condition.

    Parameters
    ----------
    condition
        A function or a nested structure of functions that takes an element of data and returns True or False.
    data
        The data to be split, which can be a nested structure (e.g., dict, list, tuple) or a single element.
    fill_None
        If True, fill the parts with None where the condition is not met.
        If False, the parts will be empty where the condition is not met.

    Returns
    -------
    data_0
        data filled only with elements that evaluate the condition to False
    data_1
        data filled only with elements that evaluate the condition to True


    """

    if isinstance(data, Mapping):
        if type(data) is not dict:
            try:
                return (None, data) if condition(data) else (data, None)
            except TypeError:
                return (data, data)
        map_0: dict[Any, Any] = {}
        map_1: dict[Any, Any] = {}
        for k, v in data.items():
            c = condition[k] if isinstance(condition, Mapping) else condition
            d0, d1 = deep_split(c, v, fill_None=fill_None)
            if d0 is not None or fill_None:
                map_0[k] = d0
            if d1 is not None or fill_None:
                map_1[k] = d1
        return map_0, map_1

    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        seq_0: list[Any] = []
        seq_1: list[Any] = []
        for i, v in enumerate(data):
            c = condition[i] if isinstance(condition, Sequence) else condition
            d0, d1 = deep_split(c, v, fill_None=fill_None)
            if d0 is not None or fill_None:
                seq_0.append(d0)
            if d1 is not None or fill_None:
                seq_1.append(d1)
        out_0: Any = seq_0
        out_1: Any = seq_1
        if isinstance(data, tuple):
            out_0 = tuple(seq_0)
            out_1 = tuple(seq_1)
        return out_0, out_1

    else:
        try:
            return (None, data) if condition(data) else (data, None)
        except TypeError:
            return (data, data)


def deep_split_by_nbytes(
    data: Any,
    max_nbytes: int,
    fill_None: bool = True,
    allow_shallow_fallback: bool = True,
) -> tuple[Any, Any]:
    """
    Recursively split data by payload size condition.

    Parameters
    ----------
    data
        The data to split.
    max_nbytes
        Maximal payload size in bytes. Elements with estimated payload smaller
        than this threshold are placed in the first output.
    fill_None
        If True, keep structure and fill missing branches with None.
        If False, remove missing branches.
    allow_shallow_fallback
        Forwarded to ``get_object_nbytes`` for unknown object types.

    Returns
    -------
    data_small
        Data filled only with elements that satisfy
        ``get_object_nbytes(element) < max_nbytes``.
    data_large
        Data filled only with elements that satisfy
        ``get_object_nbytes(element) >= max_nbytes``.


    """
    if not isinstance(max_nbytes, int) or max_nbytes < 0:
        raise ValueError(
            f"Expected non-negative integer max_nbytes, got {max_nbytes!r}"
        )

    def condition(value: Any) -> bool:
        return (
            get_object_nbytes(
                value,
                recursive=False,
                allow_shallow_fallback=allow_shallow_fallback,
            )
            < max_nbytes
        )

    data_large, data_small = deep_split(condition, data, fill_None=fill_None)
    return data_small, data_large


def deep_update(data_0: Any, data_1: Any) -> Any:
    """
    Recursively update data_0 with values from data_1.

    Parameters
    ----------
    data_0
        The original data to be updated, which can be a nested structure (e.g., dict, list, tuple) or a single element.
    data_1
        The new data to update with, which can be a nested structure (e.g., dict, list, tuple) or a single element.

    Returns
    -------
    updated_data
        The updated data after merging data_0 and data_1.


    """

    if data_0 is None:
        return data_1
    elif data_1 is None:
        return data_0
    elif type(data_0) is not type(data_1):
        raise TypeError(
            f"data_0 and data_1 must be of the same type, got {type(data_0)} and {type(data_1)}"
        )

    if isinstance(data_0, Mapping):
        updated_map: dict[Any, Any] = {}
        for k in set(data_0.keys()).union(data_1.keys()):
            v0 = data_0.get(k, None)
            v1 = data_1.get(k, None)
            updated_map[k] = deep_update(v0, v1)
        return updated_map

    elif isinstance(data_0, Sequence) and not isinstance(data_0, (str, bytes)):
        if len(data_0) != len(data_1):
            raise ValueError(
                f"data_0 and data_1 must have the same length, got {len(data_0)} and {len(data_1)}"
            )
        updated_seq: list[Any] = []
        for i in range(len(data_0)):
            v0 = data_0[i]
            v1 = data_1[i]
            updated_seq.append(deep_update(v0, v1))
        if isinstance(data_0, tuple):
            return tuple(updated_seq)
        return updated_seq

    elif isinstance(data_0, np.ndarray):
        if np.array_equal(data_0, data_1):
            return data_0
        raise ValueError(
            f"Cannot deep update non-container types: {type(data_0)} and {type(data_1)}"
        )

    try:
        if data_0 == data_1:
            return data_0
    except ValueError:
        pass

    raise ValueError(
        f"Cannot deep update non-container types: {type(data_0)} and {type(data_1)}"
    )
