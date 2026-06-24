import numpy as np
from copy import deepcopy

from foxes.core import Engine, MData
from foxes.utils import import_module
from .process import ProcessEngine, ProcessEngineRunner


dask = None
distributed = None


def delayed(func):
    """A dummy decorator"""
    return func


def load_dask():
    """On-demand loading of the dask package"""
    global dask, ProgressBar, delayed
    if dask is None:
        dask = import_module("dask")
        ProgressBar = import_module(
            "dask.diagnostics",
            pip_hint="pip install dask",
            conda_hint="conda install dask -c conda-forge",
        ).ProgressBar
        delayed = dask.delayed


def load_distributed():
    """On-demand loading of the distributed package"""
    global distributed
    if distributed is None:
        distributed = import_module("distributed")


@delayed
def _run_map(func, inputs, *args, **kwargs):
    """Helper function for running map func on proc"""
    return [func(x, *args, **kwargs) for x in inputs]


def _as_shared_local(arr):
    """Identity helper for delayed local shared-array references."""
    return arr


def _recombine_mdata_with_shared(mdata, shared):
    """Recombine chunk mdata with dask shared token data."""
    if shared is None:
        return mdata
    if shared.get("type") != "dask_shared_token":
        raise ValueError(
            "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
        )

    shared_extra_data = shared.get("extra_data")
    shared_mdata = MData(
        data=shared.get("data", {}),
        dims=shared["dims"],
        extra_data={} if shared_extra_data is None else dict(shared_extra_data),
        name=shared["name"],
        raw=True,
    )

    for name in shared_mdata.keys():
        if name in mdata:
            mdata.pop(name)
            mdata.dims.pop(name)

    mdata.recombine_with_shared(shared_mdata)
    return mdata


class DaskProcessRunner(ProcessEngineRunner):
    """Process runner variant that supports dask shared-token payloads."""

    @staticmethod
    def _resolve_shared_value(value):
        """Resolve dask future/delayed values to concrete data."""
        if hasattr(value, "result") and callable(value.result):
            return value.result()
        if hasattr(value, "compute") and callable(value.compute):
            return value.compute()
        return value

    def _recombine_mdata_with_shared(self, mdata, handle):
        """Recombine model data with either process-shm or dask token payload."""
        if handle is None:
            return mdata

        if handle.get("type") != "dask_shared_token":
            return super()._recombine_mdata_with_shared(mdata, handle)

        shared_data = {}
        for name, value in handle.get("data", {}).items():
            shared_data[name] = self._resolve_shared_value(value)

        shared_extra_data = handle.get("extra_data")
        if shared_extra_data is not None:
            shared_extra_data = self._resolve_shared_value(shared_extra_data)

        shared_mdata = MData(
            data=shared_data,
            dims=handle["dims"],
            extra_data={} if shared_extra_data is None else dict(shared_extra_data),
            name=handle["name"],
            raw=True,
        )

        for name in shared_mdata.keys():
            if name in mdata:
                mdata.pop(name)
                mdata.dims.pop(name)

        mdata.recombine_with_shared(shared_mdata)
        return mdata


class DaskEngine(ProcessEngine):
    """
    The dask engine for delayed foxes calculations.

    :group: engines

    """

    def __init__(
        self,
        *args,
        dask_config={},
        supports_shared_data=True,
        min_shared_array_bytes=0,
        progress_bar=True,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the base class
        progress_bar: bool
            Flag for showing progress bar
        kwargs: dict, optional
            Additional parameters for the base class

        """
        load_dask()
        super().__init__(
            *args,
            supports_shared_data=supports_shared_data,
            min_shared_array_bytes=min_shared_array_bytes,
            progress_bar=None,
            **kwargs,
        )
        self.dask_config = dask_config
        self._dask_progress_bar = progress_bar
        self._pbar = None

    def __enter__(self):
        if self._dask_progress_bar:
            self._pbar = ProgressBar(minimum=2)
            self._pbar.__enter__()
        dask.config.set(**self.dask_config)
        return Engine.__enter__(self)

    def __exit__(self, *args):
        if self._dask_progress_bar and self._pbar is not None:
            self._pbar.__exit__(*args)
        dask.config.refresh()
        Engine.__exit__(self, *args)

    def submit(self, f, *args, **kwargs):
        """Submits a job to worker, obtaining a future."""
        return delayed(f)(*args, **kwargs)

    def future_is_done(self, future):
        """Checks if a future is done."""
        return False

    def await_result(self, future):
        """Waits for result from a future."""
        return future.compute()

    def new_runner(self):
        """Creates a dask-aware runner for ProcessEngine chunk execution."""
        return DaskProcessRunner()

    def _print_shared_data(self, shared_mdata, verbosity):
        """Print diagnostics for data prepared as shared token payload."""
        if (
            verbosity > 1
            and shared_mdata is not None
            and (len(shared_mdata) > 0 or len(shared_mdata.extra_data) > 0)
        ):
            print(f"\n{type(self).__name__} shared data:\n")
            print(shared_mdata)
            print()

    def init_shared_memory(self, shared_memory, mdata, shared_mdata, verbosity=0):
        """Create dask shared-data token for chunk calculations."""
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        shared_data = {}
        for name, data in shared_mdata.items():
            arr = np.ascontiguousarray(data)
            ref = delayed(_as_shared_local)(arr)
            shared_data[name] = ref
            shared_memory.append(ref)

        if len(shared_data) > 0 or len(shared_mdata.extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "type": "dask_shared_token",
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": dict(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(self, mdata, shared_handle):
        """Remove entries that are restored from dask shared token in workers."""
        if shared_handle is None:
            return
        if shared_handle.get("type") != "dask_shared_token":
            raise ValueError(
                "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
            )
        for v in shared_handle.get("data", {}).keys():
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)

    def release_shared_memory(self, shared_memory, shared_handle):
        """Release dask shared-data references after chunk calculations."""
        if shared_handle is None:
            shared_memory.clear()
            return
        if shared_handle.get("type") != "dask_shared_token":
            raise ValueError(
                "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
            )
        shared_memory.clear()

    def map(
        self,
        func,
        inputs,
        *args,
        **kwargs,
    ):
        """Runs a function on a list of files."""
        if len(inputs) == 0:
            return []
        elif len(inputs) == 1:
            return [func(inputs[0], *args, **kwargs)]
        else:
            inptl = np.array_split(inputs, min(self.n_workers, len(inputs)))
            futures = []
            for subi in inptl:
                futures.append(_run_map(func, subi, *args, **kwargs))
            results = dask.compute(futures)[0]
            out = []
            for r in results:
                out += r
            return out


class LocalClusterEngine(ProcessEngine):
    """
    The dask engine for foxes calculations on a local cluster.

    Attributes
    ----------
    cluster_pars: dict
        Parameters for the cluster
    client_pars: dict
        Parameters for the client of the cluster

    :group: engines

    """

    def __init__(
        self,
        *args,
        dask_config={},
        supports_shared_data=True,
        min_shared_array_bytes=0,
        cluster_pars={},
        client_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the ProcessEngine class
        cluster_pars: dict
            Parameters for the cluster
        client_pars: dict
            Parameters for the client of the cluster
        kwargs: dict, optional
            Additional parameters for the base class

        """
        load_dask()
        super().__init__(
            *args,
            supports_shared_data=supports_shared_data,
            min_shared_array_bytes=min_shared_array_bytes,
            **kwargs,
        )

        load_distributed()

        self.cluster_pars = cluster_pars
        self.client_pars = client_pars
        self.dask_config = dask_config

        self.dask_config["scheduler"] = "distributed"
        self.dask_config["distributed.scheduler.worker-ttl"] = None

        self._cluster = None
        self._client = None

    def __enter__(self):
        self.print("Launching local dask cluster..")
        self._cluster = distributed.LocalCluster(
            n_workers=self.n_workers, **self.cluster_pars
        ).__enter__()
        self._client = distributed.Client(self._cluster, **self.client_pars).__enter__()
        self.print(self._cluster)
        self.print(f"Dashboard: {self._client.dashboard_link}\n")
        dask.config.set(**self.dask_config)
        return Engine.__enter__(self)

    def __exit__(self, *args):
        self.print(f"Shutting down {type(self._cluster).__name__}")
        # self._client.retire_workers()
        # from time import sleep
        # sleep(1)
        # self._client.shutdown()
        self._client.__exit__(*args)
        self._cluster.__exit__(*args)
        dask.config.refresh()
        Engine.__exit__(self, *args)

    def init_shared_memory(self, shared_memory, mdata, shared_mdata, verbosity=0):
        """Create dask shared-data token for chunk calculations."""
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        shared_data = {}
        for name, data in shared_mdata.items():
            arr = np.ascontiguousarray(data)
            ref = self._client.scatter(arr, broadcast=True, hash=False)
            shared_data[name] = ref
            shared_memory.append(ref)

        if len(shared_data) > 0 or len(shared_mdata.extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "type": "dask_shared_token",
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": dict(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(self, mdata, shared_handle):
        """Remove entries that are restored from dask shared token in workers."""
        if shared_handle is None:
            return
        if shared_handle.get("type") != "dask_shared_token":
            raise ValueError(
                "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
            )
        for v in shared_handle.get("data", {}).keys():
            if v in mdata:
                mdata.pop(v)
                mdata.dims.pop(v)

    def release_shared_memory(self, shared_memory, shared_handle):
        """Release dask shared-data references after chunk calculations."""
        if shared_handle is None:
            shared_memory.clear()
            return
        if shared_handle.get("type") != "dask_shared_token":
            raise ValueError(
                "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
            )
        if len(shared_memory):
            self._client.cancel(shared_memory)
        shared_memory.clear()

    def __del__(self):
        if hasattr(self, "_client") and self._client is not None:
            self._client.__del__()
        if hasattr(self, "_cluster") and self._cluster is not None:
            self._cluster.__del__()
        super().__del__()

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
        return self._client.submit(f, *args, **kwargs)

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

    def new_runner(self):
        """Creates a dask-aware runner for ProcessEngine chunk execution."""
        return DaskProcessRunner()


class SlurmClusterEngine(LocalClusterEngine):
    """
    The dask engine for foxes calculations on a SLURM cluster.

    :group: engines

    """

    def __enter__(self):
        self.print("Launching dask cluster on HPC using SLURM..")
        cargs = deepcopy(self.cluster_pars)
        nodes = cargs.pop("nodes", 1)

        dask_jobqueue = import_module(
            "dask_jobqueue",
            pip_hint="pip install setuptools dask-jobqueue",
            conda_hint="conda install setuptools dask-jobqueue -c conda-forge",
        )

        self._cluster = dask_jobqueue.SLURMCluster(**cargs)
        self._cluster.scale(futures=nodes)
        self._cluster = self._cluster.__enter__()
        self._client = distributed.Client(self._cluster, **self.client_pars).__enter__()

        self.print(self._cluster)
        self.print(f"Dashboard: {self._client.dashboard_link}\n")
        print(self._cluster.job_script())

        return LocalClusterEngine.__enter__(self)
