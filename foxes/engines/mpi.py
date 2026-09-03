from __future__ import annotations

import uuid

import numpy as np
from typing import Any

from foxes.core import MData
from foxes.utils import import_module
from foxes.utils.shared_data import (
    decode_shared_extra_data,
    encode_shared_extra_data,
)

from .process import ProcessEngine, ProcessEngineRunner


_MPI_SHARED_CACHE: dict[str, dict[str, Any]] = {}


def _mpi_create_worker_shared_cache(token: str, payload: dict[str, Any]) -> str:
    """Creates or reuses a worker-local MPI shared cache for a token."""
    if token in _MPI_SHARED_CACHE:
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

    data: dict[str, np.ndarray[Any, Any]] = {}
    extra_arrays: dict[str, np.ndarray[Any, Any]] = {}
    windows: dict[str, Any] = {}
    for target, entries in (
        (data, payload["data"]),
        (extra_arrays, payload.get("extra_arrays", {})),
    ):
        for name, meta in entries.items():
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
            shm_arr: np.ndarray[Any, Any] = np.ndarray(shape, dtype=dtype, buffer=buf)
            if rank == 0:
                shm_arr[...] = arr
            shared_comm.Barrier()

            target[name] = shm_arr
            windows[f"{id(target)}:{name}"] = win

    _MPI_SHARED_CACHE[token] = {
        "data": data,
        "dims": payload["dims"],
        "name": payload["name"],
        "extra_data": decode_shared_extra_data(
            payload.get("extra_data", {}), extra_arrays
        ),
        "shared_comm": shared_comm,
        "windows": windows,
    }

    return token


def _mpi_release_worker_shared_cache(token: str) -> str:
    """Releases worker-local MPI shared cache for a token."""
    entry = _MPI_SHARED_CACHE.pop(token, None)
    if entry is None:
        return token

    shared_comm = entry["shared_comm"]
    for win in entry["windows"].values():
        win.Free()
    shared_comm.Free()

    return token


class MPIEngineRunner(ProcessEngineRunner):
    """
    Engine runner for MPIEngine.


    """

    def _recombine_mdata_with_shared(
        self, mdata: MData, handle: dict[str, Any] | None
    ) -> MData:
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

        data = dict(cache["data"])

        shared_mdata = MData(
            data=data,
            dims=cache["dims"],
            extra_data=dict(cache.get("extra_data", {})),
            name=cache["name"],
            raw=True,
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


    """

    def new_runner(self) -> MPIEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine.

        Returns
        -------
        runner
            The engine runner

        """
        return MPIEngineRunner()

    def init_shared_memory(
        self,
        shared_memory: list[Any],
        mdata: MData,
        shared_mdata: MData | None,
        verbosity: int = 0,
    ) -> dict[str, Any] | None:
        """
        Sets the shared memory for the chunk calculation

        Parameters
        ----------
        shared_memory
            The shared memory object for the chunk calculation
        mdata
            The mdata to be used in the chunk calculation
        shared_mdata
            The shared mdata to be used in the chunk calculation
        verbosity
            The verbosity level, 0=silent

        Returns
        -------
        handle
            The handle for accessing the shared data

        """
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        token = str(uuid.uuid4())
        payload_data: dict[str, dict[str, Any]] = {}
        payload: dict[str, Any] = {
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": payload_data,
            "extra_data": dict(shared_mdata.extra_data),
        }
        for v, d in shared_mdata.items():
            assert isinstance(d, np.ndarray) and d.dtype.kind != "O", (
                f"Shared mdata entry '{v}' must be a non-object numpy array"
            )
            arr = np.ascontiguousarray(d)
            payload_data[v] = {
                "arr": arr,
                "shape": arr.shape,
                "dtype": arr.dtype.str,
            }

        extra_data, extra_arrays = encode_shared_extra_data(shared_mdata.extra_data)
        payload["extra_data"] = extra_data
        payload["extra_arrays"] = {
            name: {
                "arr": np.ascontiguousarray(data),
                "shape": data.shape,
                "dtype": data.dtype.str,
            }
            for name, data in extra_arrays.items()
        }

        if len(payload_data) or len(extra_data):
            self._print_shared_data(shared_mdata, verbosity)

        if len(payload_data) == 0 and len(extra_data) == 0:
            return None

        futures = [
            self.submit(_mpi_create_worker_shared_cache, token, payload)
            for _ in range(self.n_workers)
        ]
        for fut in futures:
            self.await_result(fut)

        return {
            "type": "mpi_shared_token",
            "token": token,
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "extra_data_keys": tuple(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(
        self, mdata: MData, shared_handle: dict[str, Any] | None
    ) -> None:
        """Remove entries that worker recombination restores from MPI shared cache."""
        if shared_handle is None:
            return

        if shared_handle.get("type") != "mpi_shared_token":
            raise ValueError(
                "MPIEngine: unsupported shared handle type, expecting 'mpi_shared_token'"
            )

        shared_vars = shared_handle.get("dims", {}).keys()
        for v in shared_vars:
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)
        self._prepare_chunk_extra_data_for_shared(mdata, shared_handle)

    def release_shared_memory(
        self,
        shared_memory: list[Any],
        shared_handle: dict[str, Any] | None,
    ) -> None:
        """
        Releases the shared memory after the chunk calculation

        Parameters
        ----------
        shared_memory
            The shared memory object for the chunk calculation
        shared_handle
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

    def _create_pool(self) -> None:
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
