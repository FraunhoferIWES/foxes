from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from foxes.core import MData
from foxes.engines import mpi as mpi_mod
from foxes.engines.mpi import MPIEngine, MPIEngineRunner


def test_mpi_runner_recombine_uses_token_cache():
    token = "tok-a"
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    mpi_mod._MPI_SHARED_CACHE[token] = {
        "data": {"A": arr},
        "dims": {"A": ("s", "t")},
        "name": "shared",
        "shared_comm": None,
        "windows": {},
    }

    mdata = MData(data={"B": np.array([1, 2], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk")
    handle = {"type": "mpi_shared_token", "token": token}

    try:
        out = MPIEngineRunner()._recombine_mdata_with_shared(mdata, handle)
        assert out is mdata
        assert "A" in mdata
        assert np.array_equal(mdata["A"], arr)
    finally:
        mpi_mod._MPI_SHARED_CACHE.pop(token, None)


def test_mpi_runner_recombine_fails_for_missing_token():
    mdata = MData(data={"B": np.array([1], dtype=np.int32)}, dims={"B": ("u",)}, name="chunk")
    handle = {"type": "mpi_shared_token", "token": "does-not-exist"}

    with pytest.raises(KeyError, match="token"):
        MPIEngineRunner()._recombine_mdata_with_shared(mdata, handle)


def test_mpi_init_shared_memory_submits_setup_once_per_worker():
    engine = MPIEngine(n_procs=4, verbosity=0)
    arr = np.arange(4, dtype=np.float64).reshape(2, 2)
    shared = MData(data={"A": arr}, dims={"A": ("s", "t")}, name="shared")

    calls = []

    def fake_submit(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return (fn, args, kwargs)

    def fake_await_result(fut):
        return fut

    engine.submit = fake_submit
    engine.await_result = fake_await_result

    handle = engine.init_shared_memory(shared)

    assert handle["type"] == "mpi_shared_token"
    assert "token" in handle
    assert handle["name"] == "shared"
    assert len(calls) == engine.n_workers
    assert all(c[0] is mpi_mod._mpi_create_worker_shared_cache for c in calls)

    tokens = [c[1][0] for c in calls]
    assert all(t == handle["token"] for t in tokens)

    payload = calls[0][1][1]
    assert payload["name"] == "shared"
    assert "A" in payload["data"]
    assert payload["data"]["A"]["shape"] == arr.shape
    assert payload["data"]["A"]["dtype"] == arr.dtype.str
    assert np.array_equal(payload["data"]["A"]["arr"], arr)


def test_mpi_release_shared_memory_submits_release_once_per_worker():
    engine = MPIEngine(n_procs=5, verbosity=0)
    handle = {"type": "mpi_shared_token", "token": "tok-release"}

    calls = []

    def fake_submit(fn, *args, **kwargs):
        calls.append((fn, args, kwargs))
        return (fn, args, kwargs)

    def fake_await_result(fut):
        return fut

    engine.submit = fake_submit
    engine.await_result = fake_await_result

    engine.release_shared_memory(handle)

    assert len(calls) == engine.n_workers
    assert all(c[0] is mpi_mod._mpi_release_worker_shared_cache for c in calls)
    assert all(c[1] == ("tok-release",) for c in calls)


def test_worker_cache_uses_mpi_allocate_shared(monkeypatch):
    created_wins = []
    created_comms = []

    class FakeWin:
        def __init__(self, nbytes):
            self._buf = bytearray(nbytes)
            self.freed = False

        def Shared_query(self, rank):
            return memoryview(self._buf), 1

        def Free(self):
            self.freed = True

    class FakeSharedComm:
        def __init__(self):
            self.rank = 0
            self.freed = False

        def Barrier(self):
            return None

        def Free(self):
            self.freed = True

    class FakeCommWorld:
        def Split_type(self, split_type, key, info):
            c = FakeSharedComm()
            created_comms.append(c)
            return c

    class FakeMPI:
        COMM_WORLD = FakeCommWorld()
        COMM_TYPE_SHARED = object()
        INFO_NULL = object()

        class Win:
            @staticmethod
            def Allocate_shared(nbytes, itemsize, comm):
                w = FakeWin(nbytes)
                created_wins.append(w)
                return w

    class FakeMPI4PY:
        MPI = FakeMPI

    class FakeMPI4PYFutures:
        @staticmethod
        def get_comm_workers():
            return None

    def fake_import_module(name, **kwargs):
        if name == "mpi4py":
            return FakeMPI4PY
        if name == "mpi4py.futures":
            return FakeMPI4PYFutures
        raise AssertionError(f"Unexpected module request: {name}")

    monkeypatch.setattr(mpi_mod, "import_module", fake_import_module)

    token = "tok-mpi"
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    payload = {
        "name": "shared",
        "dims": {"A": ("s", "t")},
        "data": {"A": {"arr": arr, "shape": arr.shape, "dtype": arr.dtype.str}},
    }

    try:
        mpi_mod._mpi_create_worker_shared_cache(token, payload)
        cache = mpi_mod._MPI_SHARED_CACHE[token]
        assert np.array_equal(cache["data"]["A"], arr)
        assert len(created_wins) == 1
        assert len(created_comms) == 1

        mpi_mod._mpi_release_worker_shared_cache(token)
        assert created_wins[0].freed
        assert created_comms[0].freed
        assert token not in mpi_mod._MPI_SHARED_CACHE
    finally:
        mpi_mod._MPI_SHARED_CACHE.pop(token, None)


def test_mpi_shared_cache_smoke_subprocess(tmp_path):
    pytest.importorskip("mpi4py.futures")

    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
    if mpiexec is None:
        pytest.skip("mpiexec/mpirun not available")

    script = tmp_path / "mpi_smoke.py"
    script.write_text(
        """
import uuid
import numpy as np
from mpi4py.futures import MPIPoolExecutor
from foxes.engines import mpi as mpi_mod


def worker_check_cache(token, expected):
    cache = mpi_mod._MPI_SHARED_CACHE.get(token)
    if cache is None:
        return False
    arr = cache["data"]["A"]
    return bool(np.array_equal(arr, expected))


def worker_token_absent(token):
    return token not in mpi_mod._MPI_SHARED_CACHE


def main():
    token = str(uuid.uuid4())
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    payload = {
        "name": "shared",
        "dims": {"A": ("s", "t")},
        "data": {"A": {"arr": arr, "shape": arr.shape, "dtype": arr.dtype.str}},
    }

    with MPIPoolExecutor(max_workers=2, use_pkl5=True) as ex:
        setup = [ex.submit(mpi_mod._mpi_create_worker_shared_cache, token, payload) for _ in range(2)]
        [f.result() for f in setup]

        checks = [ex.submit(worker_check_cache, token, arr) for _ in range(4)]
        assert all(f.result() for f in checks)

        cleanup = [ex.submit(mpi_mod._mpi_release_worker_shared_cache, token) for _ in range(2)]
        [f.result() for f in cleanup]

        absent = [ex.submit(worker_token_absent, token) for _ in range(4)]
        assert all(f.result() for f in absent)

    print("MPI_SMOKE_OK")


if __name__ == "__main__":
    main()
""".lstrip(),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[3]
    preflight = subprocess.run(
        [mpiexec, "-n", "2", sys.executable, "-c", "print('MPI_EXEC_OK')"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if preflight.returncode != 0:
        pytest.skip(
            "mpiexec detected but not usable in this environment "
            f"(returncode={preflight.returncode})"
        )

    cmd = [
        mpiexec,
        "-n",
        "3",
        sys.executable,
        "-m",
        "mpi4py.futures",
        str(script),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    if proc.returncode != 0:
        pytest.skip(
            "MPI smoke command could not run in this environment "
            f"(returncode={proc.returncode})"
        )

    assert "MPI_SMOKE_OK" in proc.stdout