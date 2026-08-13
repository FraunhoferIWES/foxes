from copy import deepcopy

import numpy as np

from foxes.utils import import_module

from foxes.core import MData
from .process import ProcessEngineRunner
from .pool import PoolEngine


ray = None


def load_ray():
    """On-demand loading of the ray package"""
    global ray
    if ray is None:
        ray = import_module("ray")


class RayEngineRunner(ProcessEngineRunner):
    """
    Engine runner for RayEngine.

    :group: engines

    """

    def _recombine_mdata_with_shared(self, mdata, handle):
        """Attach Ray object-store shared arrays to chunk-local mdata."""
        if handle is None:
            return mdata
        if handle.get("type") != "ray_shared_token":
            raise ValueError(
                "RayEngineRunner: unsupported shared handle type, expecting 'ray_shared_token'"
            )

        load_ray()
        data = {name: ray.get(ref) for name, ref in handle.get("data", {}).items()}

        shared_extra_data = handle.get("extra_data")
        shared_mdata = MData(
            data=data,
            dims=handle["dims"],
            extra_data={} if shared_extra_data is None else dict(shared_extra_data),
            name=handle["name"],
            raw=True,
        )

        mdata.recombine_with_shared(shared_mdata)
        return mdata


class RayEngine(PoolEngine):
    """
    The ray engine for foxes calculations.

    :group: engines

    """

    def __init__(self, *args, supports_shared_data=True, **kwargs):
        """Constructor."""
        super().__init__(*args, supports_shared_data=supports_shared_data, **kwargs)

    def new_runner(self):
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return RayEngineRunner()

    def init_shared_memory(self, shared_memory, mdata, shared_mdata, verbosity=0):
        """Create Ray object refs for shared chunk input data."""
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        load_ray()
        shared_data = {}
        for name, data in shared_mdata.items():
            assert isinstance(data, np.ndarray) and data.dtype.kind != "O", (
                f"Shared mdata entry '{name}' must be a non-object numpy array"
            )
            arr = np.ascontiguousarray(data)
            ref = ray.put(arr)
            shared_data[name] = ref
            shared_memory.append(ref)

        if len(shared_data) > 0 or len(shared_mdata.extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "type": "ray_shared_token",
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": dict(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(self, mdata, shared_handle):
        """Remove entries that are restored from Ray shared handle in workers."""
        if shared_handle is None:
            return
        if shared_handle.get("type") != "ray_shared_token":
            raise ValueError(
                "RayEngine: unsupported shared handle type, expecting 'ray_shared_token'"
            )

        for v in shared_handle.get("data", {}).keys():
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)

    def release_shared_memory(self, shared_memory, shared_handle):
        """Release references to Ray object-store entries created for shared data."""
        if shared_handle is None:
            shared_memory.clear()
            return
        if shared_handle.get("type") != "ray_shared_token":
            raise ValueError(
                "RayEngine: unsupported shared handle type, expecting 'ray_shared_token'"
            )

        load_ray()
        refs = list(shared_handle.get("data", {}).values())
        try:
            if len(refs) and hasattr(ray, "internal") and hasattr(ray.internal, "free"):
                ray.internal.free(refs)
        except Exception:
            pass
        shared_memory.clear()

    def _create_pool(self):
        """Creates the pool"""
        self.print(f"Initializing pool of {self.n_workers} ray workers")
        load_ray()
        ray.init(num_cpus=self.n_workers, **self.pool_args)

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

        @ray.remote
        def f_ray(*args, **kwargs):
            return f(*deepcopy(args), **deepcopy(kwargs))

        return f_ray.remote(*args, **kwargs)

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
        ready, __ = ray.wait([future])
        return len(ready) > 0

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
        return ray.get(future)

    def _shutdown_pool(self):
        """Shuts down the pool"""
        self.print(f"Shutting down pool of {self.n_workers} ray workers")
        ray.shutdown()
