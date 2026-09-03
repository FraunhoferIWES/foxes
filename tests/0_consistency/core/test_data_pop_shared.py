import numpy as np
import re
import xarray as xr

import foxes.constants as FC
from foxes.core import FData, MData


SIZE_RE = r"[0-9]+(?:\.[0-9]{2})?(?:B|KB|MB|GB|TB|PB)"


def test_pop_shared_extra_data_keeps_strict_threshold_for_arrays():
    extra_small = np.arange(8, dtype=np.int32)  # 32 bytes
    extra_equal = np.arange(16, dtype=np.int32)  # 64 bytes
    extra_large = np.arange(24, dtype=np.int32)  # 96 bytes

    mdata = MData(
        data={},
        dims={},
        extra_data={
            "extra_small": extra_small,
            "extra_equal": extra_equal,
            "extra_large": extra_large,
        },
        name="mdata",
    )

    shared = mdata.pop_shared(min_shared_array_bytes=64)

    assert np.array_equal(mdata.extra_data["extra_small"], extra_small)
    assert np.array_equal(mdata.extra_data["extra_equal"], extra_equal)
    assert "extra_large" not in mdata.extra_data

    assert "extra_small" not in shared.extra_data
    assert "extra_equal" not in shared.extra_data
    assert np.array_equal(shared.extra_data["extra_large"], extra_large)


def test_pop_shared_extra_data_keeps_nested_payloads_atomic_and_local():
    nested = {
        "inner": [
            np.arange(8, dtype=np.int32),  # 32 bytes
            np.arange(24, dtype=np.int32),  # 96 bytes
        ]
    }
    mdata = MData(data={}, dims={}, extra_data={"nested": nested}, name="mdata")

    shared = mdata.pop_shared(min_shared_array_bytes=64)

    assert "nested" in mdata.extra_data
    assert "nested" not in shared.extra_data
    assert np.array_equal(mdata.extra_data["nested"]["inner"][0], nested["inner"][0])
    assert np.array_equal(mdata.extra_data["nested"]["inner"][1], nested["inner"][1])


def test_pop_shared_extra_data_uses_xarray_dataset_payload_size():
    lookup_dataset = xr.Dataset(
        data_vars={"weights": ("point", np.arange(10_000, dtype=np.float64))}
    )
    mdata = MData(
        data={},
        dims={},
        extra_data={"gaussian_lookup": lookup_dataset},
        name="mdata",
    )

    shared = mdata.pop_shared()

    assert "gaussian_lookup" not in mdata.extra_data
    assert shared.extra_data["gaussian_lookup"] is lookup_dataset


def test_recombine_with_shared_deep_updates_extra_data():
    original = {
        "nested": {
            "inner": [
                np.arange(8, dtype=np.int32),  # 32 bytes
                np.arange(24, dtype=np.int32),  # 96 bytes
            ],
            "tag": "keep",
        }
    }
    mdata = MData(data={}, dims={}, extra_data=original, name="mdata")

    shared = mdata.pop_shared(min_shared_array_bytes=64)
    mdata.recombine_with_shared(shared)

    inner = mdata.extra_data["nested"]["inner"]
    assert len(inner) == 2
    assert inner[0].nbytes == 32
    assert inner[1].nbytes == 96
    assert mdata.extra_data["nested"]["tag"] == "keep"


def test_data_str_summarizes_without_array_payload_dump():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    mdata = MData(
        data={"A": arr},
        dims={"A": ("x", "y")},
        extra_data={
            "meta": {"a": 1},
            "vals": [1, 2, 3],
            "arr": np.arange(4, dtype=np.float64),
        },
        name="demo",
    )

    out = str(mdata)

    assert "<foxes.core.MData>" in out
    assert "<foxes.core.MData> demo" in out
    assert re.search(SIZE_RE, out)
    assert "Dimensions: (x: 2, y: 3)" in out
    assert "Coordinates:" in out
    assert "Data variables:" in out
    assert "A            (x, y)" in out
    assert "array int32 (2, 3)" in out
    assert "[0...5]" in out
    assert "x            array int64 (2,)" in out
    assert "[0...1]" in out
    assert "y            array int64 (3,)" in out
    assert "[0...2]" in out
    assert re.search(rf"A\s+\(x, y\).+{SIZE_RE}", out)
    assert "Extra data:" in out
    assert re.search(r"\n\s+meta\s+dict\(len=1\)", out)
    assert re.search(rf"\n\s+a\s+int\s+{SIZE_RE}", out)
    assert "meta.a" not in out
    assert "vals" in out
    assert "list(len=3) [1...3]" in out
    assert "arr" in out
    assert "array float64 (4,) [0.0...3.0]" in out
    assert "[[" not in out
    assert "array([" not in out


def test_from_data_copies_loop_dimensions_and_chunk_meta():
    mdata = MData(
        data={
            FC.STATE: np.array([5, 6], dtype=np.int64),
            FC.TURBINE: np.array([0, 1, 2], dtype=np.int64),
        },
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
        },
        loop_dims=[FC.STATE, FC.TURBINE],
        chunki_states=2,
        chunki_points=3,
        n_chunks_states=7,
        n_chunks_points=11,
        name="mdata",
    )

    fdata = FData.from_data(base_data=mdata, states_i0=5)

    assert np.array_equal(fdata[FC.STATE], mdata[FC.STATE])
    assert np.array_equal(fdata[FC.TURBINE], mdata[FC.TURBINE])
    assert fdata.dims[FC.STATE] == mdata.dims[FC.STATE]
    assert fdata.dims[FC.TURBINE] == mdata.dims[FC.TURBINE]
    assert fdata.sizes[FC.STATE] == mdata.sizes[FC.STATE]
    assert fdata.sizes[FC.TURBINE] == mdata.sizes[FC.TURBINE]
    assert fdata.chunki_states == mdata.chunki_states
    assert fdata.chunki_points == mdata.chunki_points
    assert fdata.n_chunks_states == mdata.n_chunks_states
    assert fdata.n_chunks_points == mdata.n_chunks_points
    assert fdata.states_i0(counter=True) == 5
