import numpy as np
import pytest
import xarray as xr

from foxes.config import config
from foxes.utils.xarray_utils import write_nc


def test_write_nc_compresses_without_rounding(tmp_path):
    source = xr.Dataset(
        data_vars={
            "value": ("row", np.arange(100, dtype=np.float64)),
            "scalar": xr.DataArray(1.0),
        },
        coords={"row": np.arange(100)},
    )
    fpath = tmp_path / "compressed.nc"

    write_nc(source, fpath, complevel=9, verbosity=0)

    with xr.open_dataset(fpath, engine=config.nc_engine) as written:
        xr.testing.assert_identical(written, source)
        assert written["value"].encoding["zlib"] is True
        assert written["value"].encoding["complevel"] == 9
        assert written["row"].encoding["zlib"] is True
        assert written["row"].encoding["complevel"] == 9
        assert written["scalar"].encoding.get("zlib", False) is False


def test_write_nc_rejects_invalid_compression_level(tmp_path):
    source = xr.Dataset({"value": ("row", np.arange(10, dtype=np.float64))})

    with pytest.raises(ValueError):
        write_nc(source, tmp_path / "invalid.nc", complevel=10, verbosity=0)


def test_write_nc_packs_without_rounding(tmp_path):
    values = np.linspace(-2.0, -1.99, 100, dtype=np.float64)
    values[50] = np.nan
    source = xr.Dataset(
        {
            "value": ("row", values),
            "integer": ("row", np.arange(100, dtype=np.int64)),
        }
    )
    fpath = tmp_path / "packed.nc"

    write_nc(source, fpath, pack=True, verbosity=0)

    with xr.open_dataset(fpath, engine=config.nc_engine) as written:
        expected = np.round(source["value"], 4)
        np.testing.assert_equal(
            np.round(written["value"].to_numpy(), 4), expected.to_numpy()
        )
        assert written["value"].encoding["dtype"] == np.dtype("int8")
        assert written["value"].encoding["zlib"] is True
        assert written["integer"].encoding["dtype"] == np.dtype("int8")
        np.testing.assert_array_equal(written["integer"], source["integer"])


def test_write_nc_applies_default_rounding_and_preserves_it_when_packed(tmp_path):
    source = xr.Dataset(
        {
            "WS": ("row", np.array([1.23456, 1.23454])),
            "WD": ("row", np.array([123.4567, 123.4564])),
        }
    )
    fpath = tmp_path / "default_rounding.nc"

    write_nc(source, fpath, pack=True, verbosity=0)

    with xr.open_dataset(fpath, engine=config.nc_engine) as written:
        np.testing.assert_equal(
            written["WS"].to_numpy(), np.round(source["WS"].to_numpy(), 4)
        )
        np.testing.assert_equal(
            written["WD"].to_numpy(), np.round(source["WD"].to_numpy(), 3)
        )


def test_write_nc_does_not_pack_values_when_default_precision_is_lost(tmp_path):
    source = xr.Dataset({"WS": ("row", np.linspace(0.0001, 30.0001, 100))})
    fpath = tmp_path / "unpacked_precision.nc"

    write_nc(source, fpath, pack=True, verbosity=0)

    with xr.open_dataset(fpath, engine=config.nc_engine) as written:
        assert written["WS"].encoding["dtype"] == np.dtype("float64")
        np.testing.assert_equal(
            written["WS"].to_numpy(), np.round(source["WS"].to_numpy(), 4)
        )


def test_write_nc_does_not_pack_coordinates(tmp_path):
    source = xr.Dataset(
        {"value": ("axis", np.linspace(0.0, 1.0, 100, dtype=np.float64))},
        coords={"axis": np.linspace(0.0, 1.0, 100, dtype=np.float64)},
    )
    fpath = tmp_path / "exact_coordinates.nc"

    write_nc(source, fpath, pack=True, verbosity=0)

    with xr.open_dataset(fpath, engine=config.nc_engine) as written:
        np.testing.assert_array_equal(written["axis"], source["axis"])
        assert written["axis"].encoding["dtype"] == np.dtype("float64")
