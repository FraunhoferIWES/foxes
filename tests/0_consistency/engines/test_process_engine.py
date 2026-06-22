import numpy as np
import pytest
from multiprocessing import shared_memory as mp_shared_memory
from multiprocessing import resource_tracker

from foxes.core import MData
from foxes.engines.process import (
    ProcessEngine,
    ProcessEngineRunner,
    _PROCESS_WORKER_SHM_CACHE,
    _install_resource_tracker_shared_memory_bypass,
    _resource_tracker_register_no_shared,
    _resource_tracker_unregister_no_shared,
)


class _DummyAlgo:
    def __init__(self):
        self.verbosity = 0
        self.n_turbines = 1
        self._chunk_store = {}

    def reset_chunk_store(self, chunk_store=None):
        if chunk_store is not None:
            self._chunk_store = chunk_store
        out = self._chunk_store
        self._chunk_store = {}
        return out


class _SharedMemoryMutatingModel:
    def calculate(self, algo, mdata, *data, **cpars):
        index = cpars["index"]
        value = cpars["value"]
        tag = cpars["tag"]
        mdata["A"][index] = value
        mdata.extra_data[tag] = value
        return {
            "written": np.array([mdata["A"][index]], dtype=np.int32),
            "local_extra": np.array([mdata.extra_data[tag]], dtype=np.int32),
        }


def _new_empty_chunk_mdata():
    return MData(
        data={"B": np.array([1, 2], dtype=np.int32)},
        dims={"B": ("u",)},
        name="chunk",
    )


def _init_shared(engine, shared):
    shared_memory = []
    handle = engine.init_shared_memory(
        shared_memory=shared_memory,
        mdata=_new_empty_chunk_mdata(),
        shared_mdata=shared,
    )
    return shared_memory, handle


def test_process_engine_resource_tracker_bypass_wrapper_behavior(monkeypatch):
    calls = []

    def _fake_register(name, rtype):
        calls.append(("register", name, rtype))
        return "register-ok"

    def _fake_unregister(name, rtype):
        calls.append(("unregister", name, rtype))
        return "unregister-ok"

    monkeypatch.setattr(
        "foxes.engines.process._resource_tracker_register", _fake_register
    )
    monkeypatch.setattr(
        "foxes.engines.process._resource_tracker_unregister", _fake_unregister
    )

    assert _resource_tracker_register_no_shared("A", "shared_memory") is None
    assert _resource_tracker_unregister_no_shared("A", "shared_memory") is None
    assert _resource_tracker_register_no_shared("A", "semaphore") == "register-ok"
    assert _resource_tracker_unregister_no_shared("A", "semaphore") == "unregister-ok"
    assert calls == [
        ("register", "A", "semaphore"),
        ("unregister", "A", "semaphore"),
    ]


def test_process_engine_resource_tracker_bypass_install_is_idempotent():
    reg_before = resource_tracker.register
    unreg_before = resource_tracker.unregister
    _install_resource_tracker_shared_memory_bypass()
    _install_resource_tracker_shared_memory_bypass()
    assert resource_tracker.register is reg_before
    assert resource_tracker.unregister is unreg_before
    assert getattr(resource_tracker, "_foxes_shm_bypass_installed", False)


def test_process_engine_init_shared_memory_roundtrip_and_release():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    shared = MData(
        data={"A": arr.copy()},
        dims={"A": ("s", "t")},
        extra_data={"source": "unit-test"},
        name="shared",
    )

    shared_memory, handle = _init_shared(engine, shared)

    assert handle is not None
    assert handle["name"] == "shared"
    assert "A" in handle["data"]
    assert dict(handle["extra_data"]) == {"source": "unit-test"}

    shm_name = handle["data"]["A"]["name"]
    shm = mp_shared_memory.SharedMemory(name=shm_name)
    try:
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        assert np.array_equal(shm_arr, arr)

        # Ensure the shared buffer is an owned copy, not aliasing the input array.
        shared["A"][0, 0] = -999.0
        assert shm_arr[0, 0] != shared["A"][0, 0]
    finally:
        shm.close()

    engine.release_shared_memory(shared_memory, handle)

    with pytest.raises(FileNotFoundError):
        mp_shared_memory.SharedMemory(name=shm_name)

    assert len(shared_memory) == 0


def test_process_engine_runner_recombine_uses_shared_memory_buffer():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    shared = MData(
        data={"A": arr.copy()},
        dims={"A": ("s", "t")},
        extra_data={"source": "unit-test"},
        name="shared",
    )
    shared_memory, handle = _init_shared(engine, shared)

    mdata = _new_empty_chunk_mdata()
    runner = ProcessEngineRunner()

    try:
        _PROCESS_WORKER_SHM_CACHE.clear()
        out = runner._recombine_mdata_with_shared(mdata, handle)
        assert out is mdata
        assert "A" in mdata
        assert np.array_equal(mdata["A"], arr)
        assert mdata.extra_data["source"] == "unit-test"

        assert not hasattr(mdata, "_shared_memory_handles")
        assert len(_PROCESS_WORKER_SHM_CACHE) == len(handle["data"])

        # Write through recombined array and verify shared segment sees the update.
        mdata["A"][1, 2] = 12345
        shm = mp_shared_memory.SharedMemory(name=handle["data"]["A"]["name"])
        try:
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            assert shm_arr[1, 2] == 12345
        finally:
            shm.close()
    finally:
        for shm in _PROCESS_WORKER_SHM_CACHE.values():
            shm.close()
        _PROCESS_WORKER_SHM_CACHE.clear()
        engine.release_shared_memory(shared_memory, handle)


def test_process_engine_runner_recombine_releases_stale_cache_handles():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    first = MData(
        data={"A": np.arange(4, dtype=np.int32).reshape(2, 2)},
        dims={"A": ("s", "t")},
        name="first",
    )
    second = MData(
        data={"A": np.arange(4, 8, dtype=np.int32).reshape(2, 2)},
        dims={"A": ("s", "t")},
        name="second",
    )

    shared_memory_a, handle_a = _init_shared(engine, first)
    shared_memory_b, handle_b = _init_shared(engine, second)

    runner = ProcessEngineRunner()
    old_names = {v["name"] for v in handle_a["data"].values()}
    new_names = {v["name"] for v in handle_b["data"].values()}

    try:
        _PROCESS_WORKER_SHM_CACHE.clear()
        mdata_a = _new_empty_chunk_mdata()
        runner._recombine_mdata_with_shared(mdata_a, handle_a)
        assert old_names.issubset(_PROCESS_WORKER_SHM_CACHE)

        mdata_b = _new_empty_chunk_mdata()
        runner._recombine_mdata_with_shared(mdata_b, handle_b)
        assert old_names.isdisjoint(_PROCESS_WORKER_SHM_CACHE)
        assert new_names.issubset(_PROCESS_WORKER_SHM_CACHE)
        assert set(_PROCESS_WORKER_SHM_CACHE) == new_names
    finally:
        for shm in _PROCESS_WORKER_SHM_CACHE.values():
            shm.close()
        _PROCESS_WORKER_SHM_CACHE.clear()
        engine.release_shared_memory(shared_memory_a, handle_a)
        engine.release_shared_memory(shared_memory_b, handle_b)


def test_process_engine_rejects_invalid_shared_entries():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    shared = MData(
        data={"BAD": np.array([{"k": 1}], dtype=object)},
        dims={"BAD": ("s",)},
        name="shared",
    )
    shared_memory = []

    with pytest.raises(AssertionError, match="must be a non-object numpy array"):
        engine.init_shared_memory(
            shared_memory=shared_memory,
            mdata=_new_empty_chunk_mdata(),
            shared_mdata=shared,
        )


def test_process_engine_pool_run_shares_memory_but_keeps_extra_data_local():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    shared = MData(
        data={"A": arr.copy()},
        dims={"A": ("s", "t")},
        extra_data={"source": "unit-test"},
        name="shared",
    )
    shared_memory, handle = _init_shared(engine, shared)
    shm_name = handle["data"]["A"]["name"]

    runner = ProcessEngineRunner()
    algo = _DummyAlgo()
    model = _SharedMemoryMutatingModel()
    mdata = _new_empty_chunk_mdata()

    try:
        engine._create_pool()
        try:
            future_a = engine.submit(
                runner.run,
                algo,
                model,
                mdata,
                shared=handle,
                chunk_store={},
                chunk_key=(0, 0),
                out_dims=("u",),
                write_nc=None,
                write_chunk_ani=None,
                utm_zone=None,
                index=(0, 0),
                value=777,
                tag="first",
            )
            future_b = engine.submit(
                runner.run,
                algo,
                model,
                mdata,
                shared=handle,
                chunk_store={},
                chunk_key=(0, 1),
                out_dims=("u",),
                write_nc=None,
                write_chunk_ani=None,
                utm_zone=None,
                index=(1, 2),
                value=888,
                tag="second",
            )

            results_a, _ = engine.await_result(future_a)
            results_b, _ = engine.await_result(future_b)

            assert results_a["written"][0] == 777
            assert results_b["written"][0] == 888
            assert results_a["local_extra"][0] == 777
            assert results_b["local_extra"][0] == 888

            shm = mp_shared_memory.SharedMemory(name=shm_name)
            try:
                shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                assert shm_arr[0, 0] == 777
                assert shm_arr[1, 2] == 888
            finally:
                shm.close()

            assert handle["extra_data"]["source"] == "unit-test"
            assert "first" not in handle["extra_data"]
            assert "second" not in handle["extra_data"]
        finally:
            engine._shutdown_pool()
    finally:
        engine.release_shared_memory(shared_memory, handle)

    with pytest.raises(FileNotFoundError):
        mp_shared_memory.SharedMemory(name=shm_name)
