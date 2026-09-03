from __future__ import annotations

import numpy as np
import atexit
from collections.abc import Sized
from multiprocessing import shared_memory as mp_shared_memory
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import resource_tracker
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.core import EngineRunner, FData, MData
from foxes.utils.shared_data import (
    decode_shared_extra_data,
    encode_shared_extra_data,
)

from .pool import PoolEngine

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, TData


_resource_tracker_register = resource_tracker.register
_resource_tracker_unregister = resource_tracker.unregister


def _resource_tracker_register_no_shared(name: str, rtype: str) -> Any:
    if rtype == "shared_memory":
        return None
    return _resource_tracker_register(name, rtype)


def _resource_tracker_unregister_no_shared(name: str, rtype: str) -> Any:
    if rtype == "shared_memory":
        return None
    return _resource_tracker_unregister(name, rtype)


def _resource_tracker_register_no_shared_typed(name: Sized, rtype: str) -> None:
    _resource_tracker_register_no_shared(str(name), rtype)


def _resource_tracker_unregister_no_shared_typed(name: Sized, rtype: str) -> None:
    _resource_tracker_unregister_no_shared(str(name), rtype)


_FOXES_SHM_BYPASS_INSTALLED = False


def _install_resource_tracker_shared_memory_bypass() -> None:
    """Installs an idempotent bypass for resource_tracker shared-memory registration.

    Shared memory ownership is managed explicitly by this engine (close/unlink in
    the parent process), so worker-side resource_tracker registration for
    ``shared_memory`` must be disabled to avoid duplicate tracking noise.
    """
    global _FOXES_SHM_BYPASS_INSTALLED
    if _FOXES_SHM_BYPASS_INSTALLED:
        return

    resource_tracker.register = _resource_tracker_register_no_shared_typed
    resource_tracker.unregister = _resource_tracker_unregister_no_shared_typed
    _FOXES_SHM_BYPASS_INSTALLED = True


_install_resource_tracker_shared_memory_bypass()


_PROCESS_WORKER_SHM_CACHE: dict[str, Any] = {}


def _close_cached_shared_memory(shm_name: str) -> None:
    """Closes and removes one cached worker shared-memory handle."""
    shm = _PROCESS_WORKER_SHM_CACHE.pop(shm_name, None)
    if shm is not None:
        try:
            shm.close()
        except FileNotFoundError:
            pass


def _close_all_cached_shared_memory() -> None:
    """Closes all cached worker shared-memory handles."""
    for shm_name in list(_PROCESS_WORKER_SHM_CACHE):
        _close_cached_shared_memory(shm_name)


atexit.register(_close_all_cached_shared_memory)


class ProcessEngineRunner(EngineRunner):
    """
    Engine runner for ProcessEngine.


    """

    def _recombine_mdata_with_shared(self, mdata: MData, handle: Any) -> MData:
        """
        Recombines the mdata with the shared data

        Parameters
        ----------
        mdata
            The mdata from the chunk calculation result
        handle
            The handle for accessing the shared data

        Returns
        -------
        recombined_mdata
            The mdata recombined with the shared data

        """
        if handle is None:
            return mdata

        shared_data = handle.get("data", {})
        shared_extra_arrays = handle.get("extra_arrays", {})
        active_shm_names = {
            value["name"]
            for entries in (shared_data, shared_extra_arrays)
            for value in entries.values()
        }
        for shm_name in list(_PROCESS_WORKER_SHM_CACHE):
            if shm_name not in active_shm_names:
                _close_cached_shared_memory(shm_name)

        data: dict[str, np.ndarray[Any, Any]] = {}
        for name, value in shared_data.items():
            shm_name = value["name"]
            shm = _PROCESS_WORKER_SHM_CACHE.get(shm_name)
            if shm is None:
                shm = mp_shared_memory.SharedMemory(name=shm_name)
                _PROCESS_WORKER_SHM_CACHE[shm_name] = shm
            data[name] = np.ndarray(
                tuple(value["shape"]),
                dtype=np.dtype(value["dtype"]),
                buffer=shm.buf,
            )

        extra_arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name, value in shared_extra_arrays.items():
            shm_name = value["name"]
            shm = _PROCESS_WORKER_SHM_CACHE.get(shm_name)
            if shm is None:
                shm = mp_shared_memory.SharedMemory(name=shm_name)
                _PROCESS_WORKER_SHM_CACHE[shm_name] = shm
            extra_arrays[name] = np.ndarray(
                tuple(value["shape"]),
                dtype=np.dtype(value["dtype"]),
                buffer=shm.buf,
            )
        shared_extra_data = decode_shared_extra_data(
            handle.get("extra_data", {}), extra_arrays
        )
        shared_mdata = MData(
            data=data,
            dims=handle["dims"],
            extra_data=shared_extra_data,
            raw=True,
            name=handle["name"],
        )

        mdata.recombine_with_shared(shared_mdata)  # modifies mdata in-place

        return mdata

    def run(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        mdata: MData,
        fdata: FData,
        tdata: TData | None = None,
        *,
        shared: Any,
        chunk_store: dict[Any, Any],
        chunk_key: Any,
        out_dims: tuple[str, ...],
        write_nc: dict[str, Any] | None,
        write_chunk_ani: dict[str, Any] | None = None,
        utm_zone: tuple[int, str] | None = None,
        **cpars: Any,
    ) -> tuple[dict[str, Any] | None, dict[Any, Any]]:
        """Helper function for running in a single process"""

        if utm_zone is not None:  # needed in some cases for mpi engine TODO investigate
            config.set_utm_zone(*utm_zone)
        algo.reset_chunk_store(chunk_store.copy())
        mdata = self._recombine_mdata_with_shared(mdata, shared)
        fdata, has_prev_farm_results = self._apply_prev_farm_results(algo, mdata, fdata)

        results: dict[str, Any] | None
        if tdata is None:
            results = model.calculate(algo, mdata, fdata, **cpars)
        else:
            results = model.calculate(algo, mdata, fdata, tdata, **cpars)
        results = self._merge_prev_farm_results(has_prev_farm_results, fdata, results)
        chunk_store = algo.reset_chunk_store()
        cstore = {chunk_key: chunk_store[chunk_key]} if chunk_key in chunk_store else {}
        self._write_ani(algo, chunk_key, write_chunk_ani, mdata, fdata, tdata)
        results = self._write_chunk_results(algo, results, write_nc, out_dims, mdata)

        return results, cstore


class ProcessEngine(PoolEngine):
    """
    The processes engine for foxes calculations.


    """

    def new_runner(self) -> ProcessEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner
            The engine runner

        """
        return ProcessEngineRunner()

    def _create_pool(self) -> None:
        """Creates the pool"""
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers, **self.pool_args)

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Submits a job to worker, obtaining a future

        Parameters
        ----------
        f
            The function f(*args, **kwargs) to be
            submitted
        args
            Arguments for the function
        kwargs
            Arguments for the function

        Returns
        -------
        future
            The future object

        """
        return self._pool.submit(f, *args, **kwargs)

    def future_is_done(self, future: Any) -> bool:
        """
        Checks if a future is done

        Parameters
        ----------
        future
            The future

        Returns
        -------
        is_done
            True if the future is done

        """
        return future.done()

    def await_result(self, future: Any) -> Any:
        """
        Waits for result from a future

        Parameters
        ----------
        future
            The future

        Returns
        -------
        result
            The calculation result

        """
        return future.result()

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

        def share_array(name: str, data: np.ndarray[Any, Any]) -> dict[str, Any]:
            assert isinstance(data, np.ndarray) and data.dtype.kind != "O", (
                f"Shared mdata entry '{name}' must be a non-object numpy array"
            )
            arr = np.ascontiguousarray(data)
            shm = mp_shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shared_memory.append({"kind": "shm", "obj": shm})

            shm_arr: np.ndarray[Any, Any] = np.ndarray(
                arr.shape, dtype=arr.dtype, buffer=shm.buf
            )
            shm_arr[...] = arr

            return {
                "name": shm.name,
                "shape": arr.shape,
                "dtype": arr.dtype.str,
            }

        shared_data = {
            name: share_array(name, data) for name, data in shared_mdata.items()
        }
        extra_data, extra_arrays = encode_shared_extra_data(shared_mdata.extra_data)
        shared_extra_arrays = {
            name: share_array(name, data) for name, data in extra_arrays.items()
        }

        if len(shared_data) > 0 or len(extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": extra_data,
            "extra_arrays": shared_extra_arrays,
            "extra_data_keys": tuple(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(
        self, mdata: MData, shared_handle: dict[str, Any]
    ) -> None:
        """Remove entries that will be restored from shared storage in workers."""
        for v in shared_handle.get("data", {}).keys():
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)
        self._prepare_chunk_extra_data_for_shared(mdata, shared_handle)

    def release_shared_memory(
        self,
        shared_memory: list[dict[str, Any]],
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
        for entry in reversed(shared_memory):
            kind = entry.get("kind")
            obj = entry.get("obj")
            if kind == "shm" and obj is not None:
                try:
                    obj.close()
                except FileNotFoundError:
                    pass
                try:
                    obj.unlink()
                except FileNotFoundError:
                    pass
            elif kind == "manager" and obj is not None:
                obj.shutdown()
        shared_memory.clear()

    def _shutdown_pool(self) -> None:
        """Shuts down the pool"""
        self._pool.shutdown()
