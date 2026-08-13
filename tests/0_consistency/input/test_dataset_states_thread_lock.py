import numpy as np

from foxes.input.states import dataset_states as dsmod


class _DummyLock:
    def __init__(self):
        self.active = False
        self.enter_count = 0

    def __enter__(self):
        self.active = True
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.active = False


class _DummyCoord:
    def __init__(self, values):
        self._values = values

    def to_numpy(self):
        return self._values


class _DummyDataset:
    def __init__(self):
        self.sizes = {"state": 3}
        self.dims = {"state": 3}
        self.coords = {"state": _DummyCoord(np.array([0, 1, 2]))}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.coords[key]
        raise KeyError(key)


class _DummyOpenContext:
    def __init__(self, dataset):
        self._dataset = dataset

    def __enter__(self):
        return self._dataset

    def __exit__(self, exc_type, exc, tb):
        return False


def test_read_nc_file_uses_lock_for_netcdf4(monkeypatch):
    lock = _DummyLock()
    monkeypatch.setattr(dsmod, "_NETCDF4_OPEN_LOCK", lock)

    def _fake_open_dataset(*args, **kwargs):
        assert kwargs["engine"] == "netcdf4"
        assert lock.active
        return _DummyOpenContext(_DummyDataset())

    monkeypatch.setattr(dsmod.xr, "open_dataset", _fake_open_dataset)

    out = dsmod._read_nc_file(
        fpath="dummy.nc",
        coords=["state"],
        vars=None,
        nc_engine="netcdf4",
        sel=None,
        isel=None,
        mode="minimal",
        drop_vars=None,
        sort=None,
        check_input_nans=False,
        preprocess=None,
    )

    assert lock.enter_count == 1
    assert np.array_equal(out, np.array([0, 1, 2]))


def test_read_nc_file_skips_lock_for_non_netcdf4(monkeypatch):
    lock = _DummyLock()
    monkeypatch.setattr(dsmod, "_NETCDF4_OPEN_LOCK", lock)

    def _fake_open_dataset(*args, **kwargs):
        assert kwargs["engine"] == "h5netcdf"
        assert not lock.active
        return _DummyOpenContext(_DummyDataset())

    monkeypatch.setattr(dsmod.xr, "open_dataset", _fake_open_dataset)

    out = dsmod._read_nc_file(
        fpath="dummy.nc",
        coords=["state"],
        vars=None,
        nc_engine="h5netcdf",
        sel=None,
        isel=None,
        mode="minimal",
        drop_vars=None,
        sort=None,
        check_input_nans=False,
        preprocess=None,
    )

    assert lock.enter_count == 0
    assert np.array_equal(out, np.array([0, 1, 2]))
