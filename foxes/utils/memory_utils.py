import numpy as np
import sys
from collections.abc import Collection, Mapping, Sequence, Set as AbstractSet

from .load import import_module


def get_object_nbytes(value, recursive=True, seen=None, allow_shallow_fallback=True):
    """Estimate payload bytes of nested extra_data values.

    The estimate is recursive for generic container types and focuses on
    payload-like elements (numpy arrays, bytes-like objects, nested
    containers), while ignoring scalar Python object overhead.

    Parameters
    ----------
    value: any
        The value to estimate the payload bytes for.
    recursive: bool
        Whether to recursively estimate the payload bytes for nested containers.
    seen: set, optional
        A set of object IDs that have already been seen to avoid double counting.
    allow_shallow_fallback: bool
        If True, use ``sys.getsizeof`` as a shallow last-resort estimate for
        unknown object types. If False, return 0 for unknown object types to
        preserve payload-only semantics.

    :group: utils

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


def print_mem(obj, min_csize=0, max_csize=None, pre_str="OBJECT SIZE"):
    """
    Prints the memory consumption of a model and its components

    Parmeters
    ---------
    obj: object
        The object to be analyzed
    min_csize: int
        The minimal size of a component for being shown
    max_csize: int, optional
        The maximal allowed size of a component
    pre_str: str
        String to be printed before

    :group: utils

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


def deep_split(condition, data, fill_None=True):
    """
    Recursively split data into two parts based on a condition.

    Parameters
    ----------
    condition: callable or nested structure of callables
        A function or a nested structure of functions that takes an element of data and returns True or False.
    data: any
        The data to be split, which can be a nested structure (e.g., dict, list, tuple) or a single element.
    fill_None: bool
        If True, fill the parts with None where the condition is not met.
        If False, the parts will be empty where the condition is not met.

    Returns
    -------
    data_0: any
        data filled only with elements that evaluate the condition to False
    data_1:
        data filled only with elements that evaluate the condition to True

    :group: utils

    """

    if isinstance(data, Mapping):
        if type(data) is not dict:
            try:
                return (None, data) if condition(data) else (data, None)
            except TypeError:
                return (data, data)
        data_0 = {}
        data_1 = {}
        for k, v in data.items():
            c = condition[k] if isinstance(condition, Mapping) else condition
            d0, d1 = deep_split(c, v, fill_None=fill_None)
            if d0 is not None or fill_None:
                data_0[k] = d0
            if d1 is not None or fill_None:
                data_1[k] = d1
        return data_0, data_1

    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        data_0 = []
        data_1 = []
        for i, v in enumerate(data):
            c = condition[i] if isinstance(condition, Sequence) else condition
            d0, d1 = deep_split(c, v, fill_None=fill_None)
            if d0 is not None or fill_None:
                data_0.append(d0)
            if d1 is not None or fill_None:
                data_1.append(d1)
        if not isinstance(data_0, type(data)):
            data_0 = type(data)(data_0)
        if not isinstance(data_1, type(data)):
            data_1 = type(data)(data_1)
        return data_0, data_1

    else:
        try:
            return (None, data) if condition(data) else (data, None)
        except TypeError:
            return (data, data)


def deep_split_by_nbytes(
    data,
    max_nbytes,
    fill_None=True,
    allow_shallow_fallback=True,
):
    """
    Recursively split data by payload size condition.

    Parameters
    ----------
    data: any
        The data to split.
    max_nbytes: int
        Maximal payload size in bytes. Elements with estimated payload smaller
        than this threshold are placed in the first output.
    fill_None: bool
        If True, keep structure and fill missing branches with None.
        If False, remove missing branches.
    allow_shallow_fallback: bool
        Forwarded to ``get_object_nbytes`` for unknown object types.

    Returns
    -------
    data_small: any
        Data filled only with elements that satisfy
        ``get_object_nbytes(element) < max_nbytes``.
    data_large: any
        Data filled only with elements that satisfy
        ``get_object_nbytes(element) >= max_nbytes``.

    :group: utils

    """
    if not isinstance(max_nbytes, int) or max_nbytes < 0:
        raise ValueError(
            f"Expected non-negative integer max_nbytes, got {max_nbytes!r}"
        )

    def condition(value):
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


def deep_update(data_0, data_1):
    """
    Recursively update data_0 with values from data_1.

    Parameters
    ----------
    data_0: any
        The original data to be updated, which can be a nested structure (e.g., dict, list, tuple) or a single element.
    data_1: any
        The new data to update with, which can be a nested structure (e.g., dict, list, tuple) or a single element.

    Returns
    -------
    updated_data: any
        The updated data after merging data_0 and data_1.

    :group: utils

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
        updated_data = {}
        for k in set(data_0.keys()).union(data_1.keys()):
            v0 = data_0.get(k, None)
            v1 = data_1.get(k, None)
            updated_data[k] = deep_update(v0, v1)
        return updated_data

    elif isinstance(data_0, Sequence) and not isinstance(data_0, (str, bytes)):
        if len(data_0) != len(data_1):
            raise ValueError(
                f"data_0 and data_1 must have the same length, got {len(data_0)} and {len(data_1)}"
            )
        updated_data = []
        for i in range(len(data_0)):
            v0 = data_0[i]
            v1 = data_1[i]
            updated_data.append(deep_update(v0, v1))
        if type(data_0) is not type(updated_data):
            updated_data = type(data_0)(updated_data)
        return updated_data

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
