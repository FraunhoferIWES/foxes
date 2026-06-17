import numpy as np
import pytest
from multiprocessing import shared_memory

from foxes.core import MData
from foxes.engines.process import ProcessEngine, ProcessEngineRunner


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
        mdata["A"][0, 0] = 777
        return {"A00": np.array([mdata["A"][0, 0]], dtype=np.int32)}


def test_process_engine_shared_memory_roundtrip_and_release():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    shared = MData(data={"A": arr.copy()}, dims={"A": ("s", "t")}, name="shared")

    handle = engine.init_shared_memory(shared)

    assert handle is not None
    assert handle["name"] == "shared"
    assert "A" in handle["data"]

    shm_name = handle["data"]["A"]["name"]
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        assert np.array_equal(shm_arr, arr)

        # Ensure the shared buffer is an owned copy, not aliasing the input array.
        shared["A"][0, 0] = -999.0
        assert shm_arr[0, 0] != shared["A"][0, 0]
    finally:
        shm.close()

    engine.release_shared_memory(handle)

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm_name)

    assert hasattr(engine, "_shared_mem")
    assert len(engine._shared_mem) == 0


def test_process_engine_runner_recombine_uses_shared_memory_buffer():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    shared = MData(data={"A": arr.copy()}, dims={"A": ("s", "t")}, name="shared")
    handle = engine.init_shared_memory(shared)

    mdata = MData(
        data={"B": np.array([1, 2], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk"
    )
    runner = ProcessEngineRunner()

    try:
        out = runner._recombine_mdata_with_shared(mdata, handle)
        assert out is mdata
        assert "A" in mdata
        assert np.array_equal(mdata["A"], arr)

        assert hasattr(mdata, "_shared_memory_handles")
        assert len(mdata._shared_memory_handles) == len(handle["data"])

        # Write through recombined array and verify shared segment sees the update.
        mdata["A"][1, 2] = 12345
        shm = shared_memory.SharedMemory(name=handle["data"]["A"]["name"])
        try:
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            assert shm_arr[1, 2] == 12345
        finally:
            shm.close()
    finally:
        # Worker-side attachment handles must be closed explicitly in this unit test.
        for shm in getattr(mdata, "_shared_memory_handles", []):
            shm.close()
        engine.release_shared_memory(handle)


def test_process_engine_rejects_invalid_shared_entries():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    shared = MData(
        data={"BAD": np.array([{"k": 1}], dtype=object)},
        dims={"BAD": ("s",)},
        name="shared",
    )

    with pytest.raises(AssertionError, match="must be a non-object numpy array"):
        engine.init_shared_memory(shared)


def test_process_engine_pool_run_shares_memory_across_processes():
    engine = ProcessEngine(n_procs=2, verbosity=0)
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    shared = MData(data={"A": arr.copy()}, dims={"A": ("s", "t")}, name="shared")
    handle = engine.init_shared_memory(shared)
    shm_name = handle["data"]["A"]["name"]

    runner = ProcessEngineRunner()
    algo = _DummyAlgo()
    model = _SharedMemoryMutatingModel()
    mdata = MData(
        data={"B": np.array([1, 2], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk"
    )

    try:
        engine._create_pool()
        try:
            future = engine.submit(
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
            )
            results, _ = engine.await_result(future)

            assert results["A00"][0] == 777

            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
                assert shm_arr[0, 0] == 777
            finally:
                shm.close()
        finally:
            engine._shutdown_pool()
    finally:
        engine.release_shared_memory(handle)
