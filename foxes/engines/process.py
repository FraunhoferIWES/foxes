import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor

from foxes.config import config
from foxes.core import EngineRunner, MData

from .pool import PoolEngine


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
            return super()._recombine_mdata_with_shared(mdata, handle)

        data = {}
        shm_handles = []
        for name, value in handle["data"].items():
            shm = shared_memory.SharedMemory(name=value["name"])
            shm_handles.append(shm)
            data[name] = np.ndarray(
                tuple(value["shape"]),
                dtype=np.dtype(value["dtype"]),
                buffer=shm.buf,
            )

        shared_mdata = MData(data=data, dims=handle["dims"], name=handle["name"])
        mdata.recombine_with_shared(shared_mdata)  # modifies mdata in-place

        # Keep SharedMemory objects alive as long as mdata is used.
        if len(shm_handles):
            handles = getattr(mdata, "_shared_memory_handles", [])
            handles += shm_handles
            setattr(mdata, "_shared_memory_handles", handles)

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

    def init_shared_memory(self, shared_mdata):
        """
        Sets the shared memory for the chunk calculation

        Parameters
        ----------
        shared_mdata: foxes.core.MData
            The shared mdata to be used in the chunk calculation

        Returns
        -------
        handle: object
            The handle for accessing the shared data

        """
        if shared_mdata is None:
            return None
        
        self._shared_mem = []
        data = {}
        for v, d in shared_mdata.items():
            assert isinstance(d, np.ndarray) and d.dtype.kind != "O" and d.nbytes, f"Shared mdata entry '{v}' must be a non-object numpy array with non-zero size"  
            arr = np.ascontiguousarray(d)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[...] = arr
            self._shared_mem.append(shm)
            data[v] = {
                "name": shm.name,
                "shape": arr.shape,
                "dtype": arr.dtype.str,
            }

        return {
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": data,
        }
    
    def release_shared_memory(self, handle):
        """
        Releases the shared memory after the chunk calculation

        Parameters
        ----------
        handle: object
            The handle for accessing the shared data

        """
        if hasattr(self, "_shared_mem"):
            while len(self._shared_mem):
                shm = self._shared_mem.pop()
                try:
                    shm.close()
                finally:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self.release_shared_memory(None)
        self._pool.shutdown()
