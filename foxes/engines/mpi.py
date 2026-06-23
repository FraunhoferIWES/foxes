import uuid
import os

import numpy as np

from foxes.core import MData
from foxes.utils import import_module

from .process import ProcessEngine, ProcessEngineRunner


_MPI_SHARED_CACHE = {}
_MPI_DEBUG_ATTACH_PRINTED = set()


def _mpi_create_worker_shared_cache(token, payload):
    """Creates or reuses a worker-local MPI shared cache for a token."""
    if token in _MPI_SHARED_CACHE:
        if payload.get("debug", False):
            cache = _MPI_SHARED_CACHE[token]
            print(
                f"MPIEngine SHM cache-hit: pid={os.getpid()} token={token} vars={len(cache['data'])}"
            )
        return token

    mpi4py = import_module(
        "mpi4py",
        pip_hint="pip install mpi4py",
        conda_hint="conda install mpi4py -c conda-forge",
    )
    mpi4py_futures = import_module(
        "mpi4py.futures",
        pip_hint="pip install mpi4py",
        conda_hint="conda install mpi4py -c conda-forge",
    )
    MPI = mpi4py.MPI

    workers_comm = mpi4py_futures.get_comm_workers()
    base_comm = MPI.COMM_WORLD if workers_comm is None else workers_comm
    shared_comm = base_comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    rank = shared_comm.rank
    dbg = payload.get("debug", False)
    dbg_rows = []

    data = {}
    windows = {}
    for name, meta in payload["data"].items():
        arr = np.ascontiguousarray(meta["arr"])
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        if arr.shape != shape or arr.dtype != dtype:
            raise ValueError(
                f"Invalid shared payload for '{name}': expected shape={shape}, dtype={dtype}, got shape={arr.shape}, dtype={arr.dtype}"
            )

        nbytes = arr.nbytes if rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, dtype.itemsize, comm=shared_comm)
        buf, _ = win.Shared_query(0)
        shm_arr = np.ndarray(shape, dtype=dtype, buffer=buf)
        if rank == 0:
            shm_arr[...] = arr
        shared_comm.Barrier()

        if dbg:
            src_ptr = int(arr.__array_interface__["data"][0])
            shm_ptr = int(shm_arr.__array_interface__["data"][0])
            copied = bool(np.shares_memory(arr, shm_arr))
            dbg_rows.append((name, src_ptr, shm_ptr, copied))

        data[name] = shm_arr
        windows[name] = win

    _MPI_SHARED_CACHE[token] = {
        "data": data,
        "dims": payload["dims"],
        "name": payload["name"],
        "extra_data": payload.get("extra_data", {}),
        "shared_comm": shared_comm,
        "windows": windows,
        "debug": dbg,
    }

    if dbg:
        world_rank = MPI.COMM_WORLD.Get_rank()
        base_rank = base_comm.Get_rank()
        ptrs = ",".join(
            [
                f"{name}:{src_ptr}->{shm_ptr}:{copied}"
                for name, src_ptr, shm_ptr, copied in dbg_rows
            ]
        )
        print(
            "MPIEngine SHM create: "
            f"pid={os.getpid()} token={token} world_rank={world_rank} "
            f"base_rank={base_rank} shared_rank={rank} vars={len(dbg_rows)} ptrs=[{ptrs}]"
        )

    return token


def _mpi_release_worker_shared_cache(token):
    """Releases worker-local MPI shared cache for a token."""
    entry = _MPI_SHARED_CACHE.pop(token, None)
    if entry is None:
        return token

    dbg = entry.get("debug", False)
    if dbg:
        print(
            f"MPIEngine SHM release-start: pid={os.getpid()} token={token} vars={len(entry['windows'])}"
        )

    _MPI_DEBUG_ATTACH_PRINTED.difference_update(
        [k for k in _MPI_DEBUG_ATTACH_PRINTED if k[1] == token]
    )

    shared_comm = entry["shared_comm"]
    for win in entry["windows"].values():
        win.Free()
    shared_comm.Free()

    if dbg:
        print(f"MPIEngine SHM release-done: pid={os.getpid()} token={token}")

    return token


class MPIEngineRunner(ProcessEngineRunner):
    """
    Engine runner for MPIEngine.

    :group: engines

    """

    def _recombine_mdata_with_shared(self, mdata, handle):
        """Attach cached MPI shared arrays to chunk-local mdata inside worker processes."""
        if handle is None:
            return mdata
        if handle.get("type") != "mpi_shared_token":
            raise ValueError(
                "MPIEngineRunner: unsupported shared handle type, expecting 'mpi_shared_token'"
            )

        token = handle["token"]
        cache = _MPI_SHARED_CACHE.get(token)
        if cache is None:
            raise KeyError(
                f"MPIEngineRunner: shared token '{token}' not found in worker cache"
            )

        if handle.get("debug", False):
            akey = (os.getpid(), token)
            if akey not in _MPI_DEBUG_ATTACH_PRINTED:
                _MPI_DEBUG_ATTACH_PRINTED.add(akey)
                ptrs = ",".join(
                    [
                        f"{name}:{int(arr.__array_interface__['data'][0])}"
                        for name, arr in cache["data"].items()
                    ]
                )
                print(
                    "MPIEngine SHM attach: "
                    f"pid={os.getpid()} token={token} vars={len(cache['data'])} ptrs=[{ptrs}]"
                )

        data = dict(cache["data"])

        shared_mdata = MData(
            data=data,
            dims=cache["dims"],
            extra_data=dict(cache.get("extra_data", {})),
            name=cache["name"],
        )
        mdata.recombine_with_shared(shared_mdata)
        return mdata


class MPIEngine(ProcessEngine):
    """
    The MPI engine for foxes calculations.

    Notes
    -----
    Builds one MPI-backed shared-memory copy of shared mdata per node and lets
    chunk tasks attach to that cache by token. This reduces repeated transfers
    to MPI workers, but still replicates the shared input once per shared-memory
    domain because MPI shared windows do not span multiple nodes.

    Examples
    --------
    Run command, e.g. for 12 processors and a script run.py:

    >>> mpiexec -n 12 -m mpi4py.futures run.py

    :group: engines

    """

    def new_runner(self):
        """
        Creates a new EngineRunner for running calculations in this engine.

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return MPIEngineRunner()

    def init_shared_memory(self, shared_memory, mdata, shared_mdata, verbosity=0):
        """
        Sets the shared memory for the chunk calculation

        Parameters
        ----------
        shared_memory: list
            The shared memory object for the chunk calculation
        mdata: foxes.core.MData
            The mdata to be used in the chunk calculation
        shared_mdata: foxes.core.MData
            The shared mdata to be used in the chunk calculation
        verbosity: int
            The verbosity level, 0=silent

        Returns
        -------
        handle: object
            The handle for accessing the shared data

        """
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        dbg = verbosity >= 2
        token = str(uuid.uuid4())
        payload = {
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": {},
            "extra_data": dict(shared_mdata.extra_data),
            "debug": dbg,
        }
        for v, d in shared_mdata.items():
            assert isinstance(d, np.ndarray) and d.dtype.kind != "O", (
                f"Shared mdata entry '{v}' must be a non-object numpy array"
            )
            arr = np.ascontiguousarray(d)
            payload["data"][v] = {
                "arr": arr,
                "shape": arr.shape,
                "dtype": arr.dtype.str,
            }

        if len(payload["data"]):
            self._print_shared_data(
                MData(
                    data={name: shared_mdata[name] for name in payload["data"]},
                    dims={name: shared_mdata.dims[name] for name in payload["data"]},
                    name=shared_mdata.name,
                ),
                verbosity,
            )

        if len(payload["data"]) == 0:
            return None

        futures = [
            self.submit(_mpi_create_worker_shared_cache, token, payload)
            for _ in range(self.n_workers)
        ]
        for fut in futures:
            self.await_result(fut)

        if dbg:
            self.print(
                f"MPIEngine SHM init-done: pid={os.getpid()} token={token} workers={self.n_workers} vars={len(payload['data'])}",
                level=2,
            )

        return {
            "type": "mpi_shared_token",
            "token": token,
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "debug": dbg,
        }

    def prepare_chunk_mdata_for_shared(self, mdata, shared_handle):
        """Remove entries that worker recombination restores from MPI shared cache."""
        if shared_handle is None:
            return

        if shared_handle.get("type") != "mpi_shared_token":
            raise ValueError(
                "MPIEngine: unsupported shared handle type, expecting 'mpi_shared_token'"
            )

        # MPI token handles do not carry a ``data`` dict like ProcessEngine;
        # shared variable names are available via the shared dims mapping.
        shared_vars = shared_handle.get("dims", {}).keys()

        for v in shared_vars:
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)

    def release_shared_memory(self, shared_memory, shared_handle):
        """
        Releases the shared memory after the chunk calculation

        Parameters
        ----------
        shared_memory: list
            The shared memory object for the chunk calculation
        shared_handle: object
            The handle for accessing the shared data

        """
        if shared_handle is None:
            shared_memory.clear()
            return
        if shared_handle.get("type") != "mpi_shared_token":
            raise ValueError(
                "MPIEngine: unsupported shared handle type, expecting 'mpi_shared_token'"
            )

        token = shared_handle["token"]
        futures = [
            self.submit(_mpi_release_worker_shared_cache, token)
            for _ in range(self.n_workers)
        ]
        for fut in futures:
            self.await_result(fut)

        shared_memory.clear()

    def _create_pool(self):
        """Creates the pool"""
        mpi4py_futures = import_module(
            "mpi4py.futures",
            pip_hint="pip install mpi4py",
            conda_hint="conda install mpi4py -c conda-forge",
        )
        MPIPoolExecutor = mpi4py_futures.MPIPoolExecutor

        pargs = dict(use_pkl5=True)
        pargs.update(self.pool_args)
        self._pool = MPIPoolExecutor(max_workers=self.n_workers, **pargs)
