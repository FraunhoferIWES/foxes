import numpy as np
import pytest
import xarray as xr

from foxes.utils.shared_data import (
    decode_shared_extra_data,
    encode_shared_extra_data,
    is_shareable_extra_data,
)


def test_shared_extra_data_dataset_roundtrip_preserves_structure():
    dataset = xr.Dataset(
        data_vars={
            "weights": (
                ("x", "y"),
                np.arange(6, dtype=np.float64).reshape(2, 3),
                {"units": "1"},
            )
        },
        coords={
            "x": ("x", np.array([1.0, 2.0])),
            "y": ("y", np.array([3, 4, 5], dtype=np.int32)),
        },
        attrs={"min_weight": 1.0e-8},
    )
    dataset.encoding = {"source": "lookup.nc"}
    dataset["weights"].encoding = {"dtype": "float64"}

    metadata, arrays = encode_shared_extra_data({"lookup": dataset})
    decoded = decode_shared_extra_data(metadata, arrays)

    xr.testing.assert_identical(decoded["lookup"], dataset)
    assert decoded["lookup"].encoding == dataset.encoding
    assert decoded["lookup"]["weights"].encoding == dataset["weights"].encoding
    weights = next(
        variable
        for variable in metadata["lookup"]["variables"]
        if variable["name"] == "weights"
    )
    assert np.shares_memory(
        decoded["lookup"]["weights"].data,
        arrays[weights["array_key"]],
    )


def test_shared_extra_data_rejects_object_arrays_for_native_sharing():
    object_array = np.array([{"value": 1}], dtype=object)
    object_dataset = xr.Dataset(data_vars={"values": ("x", object_array)})
    partly_empty_dataset = xr.Dataset(
        data_vars={
            "values": ("x", np.arange(10_000, dtype=np.float64)),
            "empty": ("empty_dim", np.array([], dtype=np.float64)),
        }
    )

    assert not is_shareable_extra_data(object_array)
    assert not is_shareable_extra_data(object_dataset)
    assert not is_shareable_extra_data(partly_empty_dataset)

    metadata, arrays = encode_shared_extra_data({"metadata": {"value": 1}})
    assert arrays == {}
    assert decode_shared_extra_data(metadata, arrays) == {"metadata": {"value": 1}}


def test_decode_shared_extra_data_rejects_unknown_descriptor():
    with pytest.raises(ValueError, match="descriptor kind"):
        decode_shared_extra_data({"value": {"kind": "unknown"}}, {})
