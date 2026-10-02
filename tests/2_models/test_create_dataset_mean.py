import numpy as np
import pytest
import xarray as xr

import foxes
from foxes.config import config
from foxes.input.states.create import create_dataset_mean
import foxes.variables as FV


def _write_source(tmp_path, directions=(350.0, 5.0, 90.0)):
    directions = np.asarray(directions, dtype=float)
    source = xr.Dataset(
        data_vars={
            FV.WS: (("time", "x"), np.full((len(directions), 1), 8.0)),
            FV.WD: (("time", "x"), directions[:, None]),
        },
        coords={"time": np.arange(len(directions)), "x": [0.0]},
    )
    source_file = tmp_path / "states.nc"
    source.to_netcdf(source_file, engine=config.nc_engine)
    return source_file


def test_create_dataset_mean_calculates_circular_main_direction(tmp_path):
    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean(
            data_source=_write_source(tmp_path),
            coord="time",
            var2ncvar={FV.WS: FV.WS, FV.WD: FV.WD},
            wd_histo_width=30.0,
            add_counts=True,
            verbosity=0,
        )

    np.testing.assert_allclose(data[FV.MAIN_WD], [357.5])
    counts = data[f"{FV.MAIN_WD}_counts"][0]
    assert counts.sizes == {"wd_bins": 24}
    assert counts[0] == 2.0
    assert counts[6] == 1.0
    assert counts.sum() == 6.0


def test_create_dataset_mean_captures_mode_across_bin_boundary(tmp_path):
    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean(
            data_source=_write_source(tmp_path, directions=(4.9, 5.1, 90.0)),
            coord="time",
            var2ncvar={FV.WS: FV.WS, FV.WD: FV.WD},
            wd_histo_width=10.0,
            verbosity=0,
        )

    np.testing.assert_allclose(data[FV.MAIN_WD], [5.0])


def test_create_dataset_mean_rejects_invalid_histogram_width(tmp_path):
    with pytest.raises(ValueError, match="bin width must be in"):
        create_dataset_mean(
            data_source=_write_source(tmp_path),
            coord="time",
            var2ncvar={FV.WS: FV.WS, FV.WD: FV.WD},
            wd_histo_width=0.0,
            verbosity=0,
        )
