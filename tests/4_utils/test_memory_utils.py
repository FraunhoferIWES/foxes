from collections import UserDict, deque

import numpy as np
import pytest
import xarray as xr

from foxes.utils.memory_utils import (
    deep_split_by_nbytes,
    deep_update,
    get_object_nbytes,
)


class _Mapping(UserDict):
    pass


def test_get_object_nbytes_supports_generic_mapping_and_sequence():
    a = np.arange(8, dtype=np.float64)
    b = np.arange(4, dtype=np.int32)
    payload = _Mapping({"a": a, "nested": [b, "txt"]})

    size = get_object_nbytes(payload)

    assert size >= a.nbytes + b.nbytes


def test_get_object_nbytes_supports_generic_collection_fallback():
    a = np.arange(6, dtype=np.float32)
    b = np.arange(2, dtype=np.int64)
    payload = deque([a, b, 123])

    size = get_object_nbytes(payload)

    assert size >= a.nbytes + b.nbytes


def test_get_object_nbytes_non_recursive_collection_uses_shallow_size():
    payload = {"a": np.arange(4, dtype=np.float64), "b": [1, 2, 3]}

    size = get_object_nbytes(payload, recursive=False)

    assert isinstance(size, int)
    assert size > 0


def test_get_object_nbytes_handles_recursive_collections():
    payload = []
    payload.append(payload)

    assert get_object_nbytes(payload) == 0


def test_get_object_nbytes_uses_direct_size_fallback():
    class _Opaque:
        pass

    obj = _Opaque()

    assert get_object_nbytes(obj) > 0


def test_get_object_nbytes_can_disable_shallow_fallback():
    class _Opaque:
        pass

    obj = _Opaque()

    assert get_object_nbytes(obj, allow_shallow_fallback=False) == 0


def test_deep_split_by_nbytes_splits_nested_data():
    a = np.arange(4, dtype=np.float64)  # 32 bytes
    b = np.arange(20, dtype=np.float64)  # 160 bytes
    payload = {"a": a, "nested": [b, b"ab"]}

    small, large = deep_split_by_nbytes(payload, max_nbytes=64)

    assert np.array_equal(small["a"], a)
    assert small["nested"][0] is None
    assert small["nested"][1] == b"ab"

    assert large["a"] is None
    assert np.array_equal(large["nested"][0], b)
    assert large["nested"][1] is None


def test_deep_split_by_nbytes_boundary_goes_to_large_partition():
    arr = np.arange(4, dtype=np.float64)  # 32 bytes

    small, large = deep_split_by_nbytes([arr], max_nbytes=arr.nbytes)

    assert small[0] is None
    assert np.array_equal(large[0], arr)


def test_deep_split_by_nbytes_rejects_invalid_max_nbytes():
    with pytest.raises(ValueError, match="max_nbytes"):
        deep_split_by_nbytes([1, 2, 3], max_nbytes=-1)

    with pytest.raises(ValueError, match="max_nbytes"):
        deep_split_by_nbytes([1, 2, 3], max_nbytes=1.5)


def test_deep_split_by_nbytes_fill_none_false_drops_missing_branches():
    a = np.arange(2, dtype=np.float64)  # 16 bytes
    b = np.arange(20, dtype=np.float64)  # 160 bytes
    payload = {"items": [a, b]}

    small, large = deep_split_by_nbytes(payload, max_nbytes=64, fill_None=False)

    assert np.array_equal(small["items"][0], a)
    assert len(small["items"]) == 1

    assert np.array_equal(large["items"][0], b)
    assert len(large["items"]) == 1


def test_deep_split_by_nbytes_respects_allow_shallow_fallback():
    class _Opaque:
        pass

    obj = _Opaque()

    small_true, large_true = deep_split_by_nbytes(
        [obj],
        max_nbytes=1,
        allow_shallow_fallback=True,
    )
    assert small_true[0] is None
    assert large_true[0] is obj

    small_false, large_false = deep_split_by_nbytes(
        [obj],
        max_nbytes=1,
        allow_shallow_fallback=False,
    )
    assert small_false[0] is obj
    assert large_false[0] is None


def test_deep_update_accepts_equal_numpy_arrays():
    arr = np.arange(5, dtype=np.int32)
    out = deep_update(arr, arr.copy())
    assert np.array_equal(out, arr)


def test_deep_split_and_update_preserve_non_dict_mapping_payloads():
    payload = {"dataset": xr.Dataset({"a": ("x", np.array([1.0, 2.0]))})}

    small, large = deep_split_by_nbytes(payload, max_nbytes=1, fill_None=True)
    merged = deep_update(small, large)

    assert isinstance(merged["dataset"], xr.Dataset)
    assert np.array_equal(merged["dataset"]["a"].to_numpy(), np.array([1.0, 2.0]))
