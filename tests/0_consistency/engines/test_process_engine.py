import numpy as np
import pytest
from multiprocessing import shared_memory as mp_shared_memory
from multiprocessing import resource_tracker
from xarray import Dataset

from foxes.core import MData
import foxes.constants as FC
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
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
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


def test_mdata_pop_shared_respects_min_size_threshold():
    shared_small = np.arange(4, dtype=np.int32)
    shared_large = np.arange(32, dtype=np.float64)
    extra_small = np.arange(4, dtype=np.int32)
    extra_large = np.arange(32, dtype=np.float64)
    nested_extra_small = {"inner": [np.arange(2, dtype=np.int32)]}
    nested_extra_large = {"inner": [np.arange(32, dtype=np.float64)]}
    mdata = MData(
        data={
            FC.STATE: np.arange(3),
            "shared_small": shared_small.copy(),
            "shared_large": shared_large.copy(),
            "chunked": np.arange(6, dtype=np.float64).reshape(3, 2),
        },
        dims={
            FC.STATE: (FC.STATE,),
            "shared_small": ("u",),
            "shared_large": ("v",),
            "chunked": (FC.STATE, "w"),
        },
        extra_data={
            "extra_small": extra_small.copy(),
            "extra_large": extra_large.copy(),
            "nested_extra_small": nested_extra_small,
            "nested_extra_large": nested_extra_large,
            "meta": {"k": 1},
        },
        loop_dims=[FC.STATE],
        name="mdata",
    )

    shared = mdata.pop_shared(min_shared_array_bytes=64)

    assert "shared_large" in shared
    assert "shared_small" not in shared
    assert "shared_small" in mdata
    assert "shared_large" not in mdata
    assert "chunked" in mdata
    assert "extra_large" in shared.extra_data
    assert "extra_small" in shared.extra_data
    assert shared.extra_data["extra_small"] is None
    assert "nested_extra_large" in shared.extra_data
    assert "nested_extra_small" in shared.extra_data
    assert shared.extra_data["nested_extra_small"] == {"inner": [None]}
    assert "extra_large" in mdata.extra_data
    assert mdata.extra_data["extra_large"] is None
    assert "extra_small" in mdata.extra_data
    assert "nested_extra_large" in mdata.extra_data
    assert mdata.extra_data["nested_extra_large"] == {"inner": [None]}
    assert "nested_extra_small" in mdata.extra_data
    assert "meta" in mdata.extra_data


def test_process_engine_get_chunk_input_data_uses_min_size_threshold_for_split():
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=1024)
    model_data = Dataset(
        coords={FC.STATE: np.arange(3), FC.TURBINE: np.arange(1)},
        data_vars={
            "shared_small": (("u",), np.arange(4, dtype=np.int32)),
            "chunked": ((FC.STATE,), np.arange(3, dtype=np.float64)),
        },
    )

    class _Algo:
        n_turbines = 1

    mdata, fdata = engine.get_chunk_input_data(
        algo=_Algo(),
        model_data=model_data,
        farm_data=None,
        point_data=None,
        states_i0_i1=(0, 3),
        targets_i0_i1=(0, 0),
        out_vars=[],
        chunki_states=0,
        chunki_points=0,
        n_chunks_states=1,
        n_chunks_points=1,
    )

    assert "shared_small" in mdata
    assert "chunked" in mdata
    assert FC.STATE in fdata


def test_process_engine_init_shared_memory_respects_min_size_threshold():
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=64)
    small = np.arange(4, dtype=np.int32)
    large = np.arange(24, dtype=np.float64)
    source = MData(
        data={"small": small.copy(), "large": large.copy()},
        dims={"small": ("s",), "large": ("t",)},
        name="shared",
    )
    shared = source.pop_shared(min_shared_array_bytes=64)

    shared_memory, handle = _init_shared(engine, shared)

    assert handle is not None
    assert "large" in handle["data"]
    assert "small" not in handle["data"]
    assert "local_data" not in handle

    shm_entries = [entry for entry in shared_memory if entry["kind"] == "shm"]
    assert len(shm_entries) >= 1

    shm_name = handle["data"]["large"]["name"]
    shm = mp_shared_memory.SharedMemory(name=shm_name)
    try:
        shm_arr = np.ndarray(large.shape, dtype=large.dtype, buffer=shm.buf)
        assert np.array_equal(shm_arr, large)
    finally:
        shm.close()

    engine.release_shared_memory(shared_memory, handle)

    with pytest.raises(FileNotFoundError):
        mp_shared_memory.SharedMemory(name=shm_name)


def test_process_engine_runner_recombine_keeps_chunk_data_when_nothing_is_shared():
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=1024)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    source = MData(
        data={"A": arr.copy()},
        dims={"A": ("s", "t")},
        name="shared",
    )
    shared = source.pop_shared(min_shared_array_bytes=1024)

    shared_memory, handle = _init_shared(engine, shared)
    mdata = _new_empty_chunk_mdata()
    runner = ProcessEngineRunner()

    try:
        _PROCESS_WORKER_SHM_CACHE.clear()
        out = runner._recombine_mdata_with_shared(mdata, handle)
        assert out is mdata
        assert "A" not in mdata
        assert handle is None
        assert len(_PROCESS_WORKER_SHM_CACHE) == 0
        assert not any(entry["kind"] == "shm" for entry in shared_memory)
    finally:
        _PROCESS_WORKER_SHM_CACHE.clear()
        engine.release_shared_memory(shared_memory, handle)


def test_process_engine_prepare_chunk_mdata_for_shared_removes_shared_keys():
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
    mdata = MData(
        data={
            "A": np.array([1, 2], dtype=np.int32),
            "B": np.array([3, 4], dtype=np.int32),
        },
        dims={"A": ("u",), "B": ("u",)},
        name="chunk",
    )
    shared_handle = {
        "data": {
            "A": {"name": "unused", "shape": (2,), "dtype": np.dtype(np.int32).str}
        }
    }

    engine.prepare_chunk_mdata_for_shared(mdata, shared_handle)

    assert "A" not in mdata
    assert "A" not in mdata.dims
    assert "B" in mdata
    assert "B" in mdata.dims


def test_process_engine_does_not_print_shared_data_when_nothing_is_shared():
    engine = ProcessEngine(n_procs=2, verbosity=2, min_shared_array_bytes=1024)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    source = MData(
        data={"A": arr.copy()},
        dims={"A": ("s", "t")},
        name="shared",
    )
    shared = source.pop_shared(min_shared_array_bytes=1024)

    calls = []

    def fake_print(shared_mdata, verbosity):
        calls.append((shared_mdata, verbosity))

    engine._print_shared_data = fake_print

    shared_memory, handle = _init_shared(engine, shared)
    try:
        assert handle is None
        assert calls == []
    finally:
        engine.release_shared_memory(shared_memory, handle)


def test_process_engine_runner_recombine_uses_shared_memory_buffer():
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
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
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
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
    engine = ProcessEngine(n_procs=2, verbosity=0, min_shared_array_bytes=0)
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
