import numpy as np
import pytest

from foxes.core import MData
from foxes.engines import ray as ray_mod
from foxes.engines.ray import RayEngine, RayEngineRunner


class _FakeRayInternal:
    def __init__(self):
        self.freed = []

    def free(self, refs):
        self.freed.append(list(refs))


class _FakeRemoteFunc:
    def __init__(self, func):
        self.func = func

    def remote(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class _FakeRay:
    def __init__(self):
        self._store = {}
        self._counter = 0
        self.internal = _FakeRayInternal()

    def put(self, obj):
        key = f"ref-{self._counter}"
        self._counter += 1
        self._store[key] = np.array(obj, copy=True)
        return key

    def get(self, ref):
        if isinstance(ref, list):
            return [self._store[r] for r in ref]
        return self._store.get(ref, ref)

    def wait(self, futures):
        return futures, []

    def remote(self, f):
        return _FakeRemoteFunc(f)

    def init(self, **kwargs):
        return None

    def shutdown(self):
        return None


def _install_fake_ray(monkeypatch):
    fake = _FakeRay()
    monkeypatch.setattr(ray_mod, "ray", fake)
    monkeypatch.setattr(ray_mod, "load_ray", lambda: None)
    return fake


def test_ray_init_shared_memory_returns_token_handle(monkeypatch):
    fake = _install_fake_ray(monkeypatch)
    engine = RayEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
    shared = MData(
        data={"A": np.arange(6, dtype=np.int32).reshape(2, 3)},
        dims={"A": ("s", "t")},
        extra_data={"source": "unit-test"},
        name="shared",
    )

    shared_memory = []
    handle = engine.init_shared_memory(shared_memory, MData(name="chunk"), shared)

    assert handle["type"] == "ray_shared_token"
    assert handle["name"] == "shared"
    assert "A" in handle["data"]
    assert handle["extra_data"] == {"source": "unit-test"}
    assert handle["data"]["A"] in shared_memory
    assert len(shared_memory) >= 1
    assert np.array_equal(fake.get(handle["data"]["A"]), shared["A"])


def test_ray_runner_recombine_uses_ray_refs(monkeypatch):
    fake = _install_fake_ray(monkeypatch)
    arr = np.arange(4, dtype=np.float64)
    ref = fake.put(arr)
    mdata = MData(
        data={"B": np.array([1], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk"
    )
    handle = {
        "type": "ray_shared_token",
        "name": "shared",
        "dims": {"A": ("s",)},
        "data": {"A": ref},
        "extra_data": {"source": "unit-test"},
    }

    out = RayEngineRunner()._recombine_mdata_with_shared(mdata, handle)

    assert out is mdata
    assert np.array_equal(mdata["A"], arr)
    assert mdata.extra_data["source"] == "unit-test"


def test_ray_runner_recombine_rejects_non_token_handle(monkeypatch):
    _install_fake_ray(monkeypatch)
    mdata = MData(
        data={"B": np.array([1], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk"
    )

    with pytest.raises(ValueError, match="ray_shared_token"):
        RayEngineRunner()._recombine_mdata_with_shared(mdata, {"type": "legacy"})


def test_ray_prepare_chunk_mdata_for_shared_removes_shared_keys(monkeypatch):
    _install_fake_ray(monkeypatch)
    engine = RayEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
    mdata = MData(
        data={
            "A": np.array([1, 2], dtype=np.int32),
            "B": np.array([3, 4], dtype=np.int32),
        },
        dims={"A": ("u",), "B": ("u",)},
        name="chunk",
    )

    engine.prepare_chunk_mdata_for_shared(
        mdata,
        {"type": "ray_shared_token", "data": {"A": "ref-0"}},
    )

    assert "A" not in mdata
    assert "A" not in mdata.dims
    assert "B" in mdata


def test_ray_release_shared_memory_frees_refs(monkeypatch):
    fake = _install_fake_ray(monkeypatch)
    engine = RayEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
    shared_memory = ["ref-0", "ref-1"]

    engine.release_shared_memory(
        shared_memory,
        {"type": "ray_shared_token", "data": {"A": "ref-0", "B": "ref-1"}},
    )

    assert fake.internal.freed[-1] == ["ref-0", "ref-1"]
    assert shared_memory == []


def test_ray_release_shared_memory_rejects_non_token_handle(monkeypatch):
    _install_fake_ray(monkeypatch)
    engine = RayEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)

    with pytest.raises(ValueError, match="ray_shared_token"):
        engine.release_shared_memory([], {"type": "legacy"})
