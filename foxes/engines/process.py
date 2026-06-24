import numpy as np
import atexit
from multiprocessing import Manager, shared_memory as mp_shared_memory
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import resource_tracker

from foxes.config import config
from foxes.core import EngineRunner, MData

from .pool import PoolEngine


_resource_tracker_register = resource_tracker.register
_resource_tracker_unregister = resource_tracker.unregister


def _resource_tracker_register_no_shared(name, rtype):
    if rtype == "shared_memory":
        return
    return _resource_tracker_register(name, rtype)


def _resource_tracker_unregister_no_shared(name, rtype):
    if rtype == "shared_memory":
        return
    return _resource_tracker_unregister(name, rtype)


def _install_resource_tracker_shared_memory_bypass():
    """Installs an idempotent bypass for resource_tracker shared-memory registration.

    Shared memory ownership is managed explicitly by this engine (close/unlink in
    the parent process), so worker-side resource_tracker registration for
    ``shared_memory`` must be disabled to avoid duplicate tracking noise.
    """
    if getattr(resource_tracker, "_foxes_shm_bypass_installed", False):
        return

    resource_tracker.register = _resource_tracker_register_no_shared
    resource_tracker.unregister = _resource_tracker_unregister_no_shared
    resource_tracker._foxes_shm_bypass_installed = True


_install_resource_tracker_shared_memory_bypass()


_PROCESS_WORKER_SHM_CACHE = {}


def _close_cached_shared_memory(shm_name):
    """Closes and removes one cached worker shared-memory handle."""
    shm = _PROCESS_WORKER_SHM_CACHE.pop(shm_name, None)
    if shm is not None:
        try:
            shm.close()
        except FileNotFoundError:
            pass


def _close_all_cached_shared_memory():
    """Closes all cached worker shared-memory handles."""
    for shm_name in list(_PROCESS_WORKER_SHM_CACHE):
        _close_cached_shared_memory(shm_name)


atexit.register(_close_all_cached_shared_memory)


class ProcessEngineRunner(EngineRunner):
    """
    Engine runner for ProcessEngine.

    :group: engines

    """

    def _recombine_mdata_with_shared(self, mdata, handle):
        """
        Recombines the mdata with the shared data

        Parameters
        ----------
        mdata: foxes.core.MData
            The mdata from the chunk calculation result
        handle: object
            The handle for accessing the shared data

        Returns
        -------
        recombined_mdata: foxes.core.MData
            The mdata recombined with the shared data

        """
        if handle is None:
            return mdata

        shared_data = handle.get("data", {})
        active_shm_names = {value["name"] for value in shared_data.values()}
        for shm_name in list(_PROCESS_WORKER_SHM_CACHE):
            if shm_name not in active_shm_names:
                _close_cached_shared_memory(shm_name)

        data = {}
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

        shared_extra_data = handle.get("extra_data")
        shared_mdata = MData(
            data=data,
            dims=handle["dims"],
            extra_data={} if shared_extra_data is None else dict(shared_extra_data),
            raw=True,
            name=handle["name"],
        )

        mdata.recombine_with_shared(shared_mdata)  # modifies mdata in-place

        return mdata

    def run(
        self,
        algo,
        model,
        mdata,
        *data,
        shared,
        chunk_store,
        chunk_key,
        out_dims,
        write_nc,
        write_chunk_ani=None,
        utm_zone=None,
        **cpars,
    ):
        """Helper function for running in a single process"""

        if utm_zone is not None:  # needed in some cases for mpi engine TODO investigate
            config.set_utm_zone(*utm_zone)
        algo.reset_chunk_store(chunk_store.copy())
        mdata = self._recombine_mdata_with_shared(mdata, shared)

        results = model.calculate(algo, mdata, *data, **cpars)
        chunk_store = algo.reset_chunk_store()
        cstore = {chunk_key: chunk_store[chunk_key]} if chunk_key in chunk_store else {}
        self._write_ani(algo, chunk_key, write_chunk_ani, mdata, *data)
        results = self._write_chunk_results(algo, results, write_nc, out_dims, mdata)
        return results, cstore


class ProcessEngine(PoolEngine):
    """
    The processes engine for foxes calculations.

    :group: engines

    """

    def __init__(
        self,
        *args,
        supports_shared_data=True,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Positional arguments forwarded to ``PoolEngine`` / base classes.
        supports_shared_data: bool
            Flag forwarded to ``PoolEngine`` indicating whether shared-data
            preparation should be used. Defaults to ``True``.
        kwargs: dict, optional
            Additional keyword arguments forwarded to ``PoolEngine`` / base
            classes.

        """
        super().__init__(*args, supports_shared_data=supports_shared_data, **kwargs)

    def new_runner(self):
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return ProcessEngineRunner()

    def _create_pool(self):
        """Creates the pool"""
        self._pool = ProcessPoolExecutor(max_workers=self.n_workers, **self.pool_args)

    def submit(self, f, *args, **kwargs):
        """
        Submits a job to worker, obtaining a future

        Parameters
        ----------
        f: Callable
            The function f(*args, **kwargs) to be
            submitted
        args: tuple, optional
            Arguments for the function
        kwargs: dict, optional
            Arguments for the function

        Returns
        -------
        future: object
            The future object

        """
        return self._pool.submit(f, *args, **kwargs)

    def future_is_done(self, future):
        """
        Checks if a future is done

        Parameters
        ----------
        future: object
            The future

        Returns
        -------
        is_done: bool
            True if the future is done

        """
        return future.done()

    def await_result(self, future):
        """
        Waits for result from a future

        Parameters
        ----------
        future: object
            The future

        Returns
        -------
        result: object
            The calculation result

        """
        return future.result()

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

        shared_data = {}
        for name, data in shared_mdata.items():
            assert isinstance(data, np.ndarray) and data.dtype.kind != "O", (
                f"Shared mdata entry '{name}' must be a non-object numpy array"
            )
            arr = np.ascontiguousarray(data)
            shm = mp_shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shared_memory.append({"kind": "shm", "obj": shm})

            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[...] = arr

            shared_data[name] = {
                "name": shm.name,
                "shape": arr.shape,
                "dtype": arr.dtype.str,
            }

        extra_data = None
        if len(shared_mdata.extra_data):
            manager = Manager()
            shared_memory.append({"kind": "manager", "obj": manager})
            extra_data = manager.dict(shared_mdata.extra_data)

        if len(shared_data) > 0 or extra_data is not None:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": extra_data,
        }

    def prepare_chunk_mdata_for_shared(self, mdata, shared_handle):
        """Remove entries that will be restored from shared storage in workers."""
        for v in shared_handle.get("data", {}).keys():
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
        for entry in reversed(shared_memory):
            kind = entry.get("kind")
            obj = entry.get("obj")
            if kind == "shm":
                try:
                    obj.close()
                except FileNotFoundError:
                    pass
                try:
                    obj.unlink()
                except FileNotFoundError:
                    pass
            elif kind == "manager":
                obj.shutdown()
        shared_memory.clear()

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self._pool.shutdown()
