from __future__ import annotations

import numpy as np
import xarray as xr
from copy import deepcopy
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any

from foxes.core import Engine, MData
from foxes.config import config
from foxes.utils import import_module
from foxes.utils.shared_data import (
    decode_shared_extra_data,
    encode_shared_extra_data,
)
from .process import ProcessEngine, ProcessEngineRunner

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, FData, TData


dask = None
distributed = None
ProgressBar: Any = None


def delayed(func: Any) -> Any:
    """A dummy decorator"""
    return func


def load_dask() -> None:
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


def load_distributed() -> None:
    """On-demand loading of the distributed package"""
    global distributed
    if distributed is None:
        distributed = import_module("distributed")


def _get_dask() -> Any:
    """Get loaded dask module with non-None type for static checkers."""
    load_dask()
    assert dask is not None
    return dask


def _get_distributed() -> Any:
    """Get loaded distributed module with non-None type for static checkers."""
    load_distributed()
    assert distributed is not None
    return distributed


def _get_progress_bar_cls() -> Any:
    """Get ProgressBar class with non-None type for static checkers."""
    load_dask()
    assert ProgressBar is not None
    return ProgressBar


@delayed
def _run_map(func: Any, inputs: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Helper function for running map func on proc"""
    return [func(x, *args, **kwargs) for x in inputs]


def _as_shared_local(arr: np.ndarray) -> np.ndarray:
    """Identity helper for delayed local shared-array references."""
    return arr


def _recombine_mdata_with_shared(mdata: MData, shared: dict[str, Any] | None) -> MData:
    """Recombine chunk mdata with dask shared token data."""
    if shared is None:
        return mdata
    if shared.get("type") != "dask_shared_token":
        raise ValueError(
            "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
        )

    shared_extra_data = decode_shared_extra_data(
        shared.get("extra_data", {}), shared.get("extra_arrays", {})
    )
    shared_mdata = MData(
        data=shared.get("data", {}),
        dims=shared["dims"],
        extra_data=shared_extra_data,
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
    """
    Process runner variant that supports dask shared-token payloads.
    """

    @staticmethod
    def _resolve_shared_value(value: Any) -> Any:
        """Resolve dask future/delayed values to concrete data."""
        if hasattr(value, "result") and callable(value.result):
            return value.result()
        if hasattr(value, "compute") and callable(value.compute):
            return value.compute()
        return value

    @classmethod
    def _resolve_nested_value(cls, value: Any) -> Any:
        """Resolve future-like values recursively in nested containers."""
        if isinstance(value, (xr.Dataset, xr.DataArray)):
            return value
        value = cls._resolve_shared_value(value)
        if isinstance(value, Mapping):
            return {k: cls._resolve_nested_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._resolve_nested_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._resolve_nested_value(v) for v in value)
        return value

    @classmethod
    def _resolve_data_container(cls, data: Any) -> Any:
        """Resolve future-like values inside Data/MData containers in-place."""
        if data is None:
            return None

        if isinstance(data, MutableMapping):
            for k in list(data.keys()):
                data[k] = cls._resolve_nested_value(data[k])

        if hasattr(data, "extra_data"):
            data.extra_data = cls._resolve_nested_value(data.extra_data)

        return data

    def _recombine_mdata_with_shared(
        self, mdata: MData, handle: dict[str, Any] | None
    ) -> MData:
        """Recombine model data with either process-shm or dask token payload."""
        if handle is None:
            return mdata

        if handle.get("type") != "dask_shared_token":
            return super()._recombine_mdata_with_shared(mdata, handle)

        shared_data = {}
        for name, value in handle.get("data", {}).items():
            shared_data[name] = self._resolve_shared_value(value)

        extra_arrays = {
            name: self._resolve_shared_value(value)
            for name, value in handle.get("extra_arrays", {}).items()
        }
        shared_extra_data = decode_shared_extra_data(
            handle.get("extra_data", {}), extra_arrays
        )

        shared_mdata = MData(
            data=shared_data,
            dims=handle["dims"],
            extra_data=shared_extra_data,
            name=handle["name"],
            raw=True,
        )

        for name in shared_mdata.keys():
            if name in mdata:
                mdata.pop(name)
                mdata.dims.pop(name)

        mdata.recombine_with_shared(shared_mdata)
        return mdata

    def run(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        mdata: MData,
        fdata: FData,
        tdata: TData | None = None,
        *,
        shared: dict[str, Any] | None,
        chunk_store: dict[Any, Any],
        chunk_key: Any,
        out_dims: tuple[str, ...],
        write_nc: dict[str, Any] | None,
        write_chunk_ani: dict[str, Any] | None = None,
        utm_zone: tuple[int, str] | None = None,
        **cpars: Any,
    ) -> tuple[dict[str, Any] | None, dict[Any, Any]]:
        """Run one chunk while resolving future-like payloads before calculate."""
        if utm_zone is not None:
            config.set_utm_zone(*utm_zone)

        algo.reset_chunk_store(chunk_store.copy())
        mdata = self._recombine_mdata_with_shared(mdata, shared)
        mdata = self._resolve_data_container(mdata)
        fdata = self._resolve_data_container(fdata)
        tdata = self._resolve_data_container(tdata)
        cpars = self._resolve_nested_value(cpars)
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


class DaskEngine(ProcessEngine):
    """
    The dask engine for delayed foxes calculations.
    """

    def __init__(
        self,
        *args: Any,
        dask_config: dict[str, Any] | None = None,
        supports_shared_data: bool = True,
        min_shared_array_bytes: int = 0,
        progress_bar: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional parameters for the base class
        progress_bar
            Flag for showing progress bar
        kwargs
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
        self.dask_config = {} if dask_config is None else dask_config
        self._dask_progress_bar = progress_bar
        self._pbar: Any | None = None

    def __enter__(self) -> DaskEngine:
        dask_mod = _get_dask()
        if self._dask_progress_bar:
            progress_bar_cls = _get_progress_bar_cls()
            self._pbar = progress_bar_cls(minimum=2)
            self._pbar.__enter__()
        dask_mod.config.set(**self.dask_config)
        Engine.__enter__(self)
        return self

    def __exit__(self, *args: Any) -> None:
        dask_mod = _get_dask()
        if self._dask_progress_bar and self._pbar is not None:
            self._pbar.__exit__(*args)
        dask_mod.config.refresh()
        Engine.__exit__(self, *args)

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> Any:
        """Submits a job to worker, obtaining a future."""
        return delayed(f)(*args, **kwargs)

    def future_is_done(self, future: Any) -> bool:
        """Checks if a future is done."""
        return False

    def await_result(self, future: Any) -> Any:
        """Waits for result from a future."""
        return future.compute()

    def new_runner(self) -> DaskProcessRunner:
        """Creates a dask-aware runner for ProcessEngine chunk execution."""
        return DaskProcessRunner()

    def _print_shared_data(self, shared_mdata: Any, verbosity: int) -> None:
        """Print diagnostics for data prepared as shared token payload."""
        if (
            verbosity > 1
            and shared_mdata is not None
            and (len(shared_mdata) > 0 or len(shared_mdata.extra_data) > 0)
        ):
            print(f"\n{type(self).__name__} shared data:\n")
            print(shared_mdata)
            print()

    def init_shared_memory(
        self,
        shared_memory: list[Any],
        mdata: MData,
        shared_mdata: MData | None,
        verbosity: int = 0,
    ) -> dict[str, Any] | None:
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

        extra_data, extra_arrays = encode_shared_extra_data(shared_mdata.extra_data)
        shared_extra_arrays = {}
        for name, data in extra_arrays.items():
            ref = delayed(_as_shared_local)(np.ascontiguousarray(data))
            shared_extra_arrays[name] = ref
            shared_memory.append(ref)

        if len(shared_data) > 0 or len(shared_mdata.extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "type": "dask_shared_token",
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": extra_data,
            "extra_arrays": shared_extra_arrays,
            "extra_data_keys": tuple(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(
        self, mdata: MData, shared_handle: dict[str, Any] | None
    ) -> None:
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
        self._prepare_chunk_extra_data_for_shared(mdata, shared_handle)

    def release_shared_memory(
        self,
        shared_memory: list[Any],
        shared_handle: dict[str, Any] | None,
    ) -> None:
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
        func: Any,
        inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Runs a function on a list of files."""
        if len(inputs) == 0:
            return []
        elif len(inputs) == 1:
            return [func(inputs[0], *args, **kwargs)]
        else:
            dask_mod = _get_dask()
            inptl = np.array_split(inputs, min(self.n_workers, len(inputs)))
            futures = []
            for subi in inptl:
                futures.append(_run_map(func, subi, *args, **kwargs))
            results = dask_mod.compute(futures)[0]
            out = []
            for r in results:
                out += r
            return out


class LocalClusterEngine(ProcessEngine):
    """
    The dask engine for foxes calculations on a local cluster.
    """

    def __init__(
        self,
        *args: Any,
        dask_config: dict[str, Any] | None = None,
        supports_shared_data: bool = True,
        min_shared_array_bytes: int = 0,
        min_submit_array_bytes: int = 0,  # 1048576,
        cluster_pars: dict[str, Any] | None = None,
        client_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Additional parameters for the ProcessEngine class
        cluster_pars
            Parameters for the cluster
        client_pars
            Parameters for the client of the cluster
        min_submit_array_bytes
            Minimum ndarray size in bytes to scatter as dask futures before
            task submission. Values <= 0 disable this conversion.
        kwargs
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

        self.cluster_pars = {} if cluster_pars is None else cluster_pars
        self.client_pars = {} if client_pars is None else client_pars
        self.dask_config = {} if dask_config is None else dask_config
        self.min_submit_array_bytes = int(min_submit_array_bytes)

        self.dask_config["scheduler"] = "distributed"
        self.dask_config["distributed.scheduler.worker-ttl"] = None

        self._cluster: Any | None = None
        self._client: Any | None = None

    def __enter__(self) -> LocalClusterEngine:
        distributed_mod = _get_distributed()
        dask_mod = _get_dask()
        self.print("Launching local dask cluster..")
        self._cluster = distributed_mod.LocalCluster(
            n_workers=self.n_workers, **self.cluster_pars
        ).__enter__()
        self._client = distributed_mod.Client(
            self._cluster, **self.client_pars
        ).__enter__()
        self.print(self._cluster)
        client = self._client
        assert client is not None
        self.print(f"Dashboard: {client.dashboard_link}\n")
        dask_mod.config.set(**self.dask_config)
        Engine.__enter__(self)
        return self

    def __exit__(self, *args: Any) -> None:
        dask_mod = _get_dask()
        self.print(f"Shutting down {type(self._cluster).__name__}")
        # self._client.retire_workers()
        # from time import sleep
        # sleep(1)
        # self._client.shutdown()
        client = self._client
        cluster = self._cluster
        if client is not None:
            client.__exit__(*args)
        if cluster is not None:
            cluster.__exit__(*args)
        dask_mod.config.refresh()
        Engine.__exit__(self, *args)

    def init_shared_memory(
        self,
        shared_memory: list[Any],
        mdata: MData,
        shared_mdata: MData | None,
        verbosity: int = 0,
    ) -> dict[str, Any] | None:
        """Create dask shared-data token for chunk calculations."""
        if shared_mdata is None or (
            len(shared_mdata) == 0 and len(shared_mdata.extra_data) == 0
        ):
            return None

        shared_data = {}
        client = self._client
        assert client is not None
        for name, data in shared_mdata.items():
            arr = np.ascontiguousarray(data)
            ref = client.scatter(arr, broadcast=True, hash=False)
            shared_data[name] = ref
            shared_memory.append(ref)

        extra_data, extra_arrays = encode_shared_extra_data(shared_mdata.extra_data)
        shared_extra_arrays = {}
        for name, data in extra_arrays.items():
            ref = client.scatter(np.ascontiguousarray(data), broadcast=True, hash=False)
            shared_extra_arrays[name] = ref
            shared_memory.append(ref)

        if len(shared_data) > 0 or len(shared_mdata.extra_data) > 0:
            self._print_shared_data(shared_mdata, verbosity=verbosity)

        return {
            "type": "dask_shared_token",
            "name": shared_mdata.name,
            "dims": shared_mdata.dims,
            "data": shared_data,
            "extra_data": extra_data,
            "extra_arrays": shared_extra_arrays,
            "extra_data_keys": tuple(shared_mdata.extra_data),
        }

    def prepare_chunk_mdata_for_shared(
        self, mdata: MData, shared_handle: dict[str, Any] | None
    ) -> None:
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
        self._prepare_chunk_extra_data_for_shared(mdata, shared_handle)

    def release_shared_memory(
        self,
        shared_memory: list[Any],
        shared_handle: dict[str, Any] | None,
    ) -> None:
        """Release dask shared-data references after chunk calculations."""
        if shared_handle is None:
            shared_memory.clear()
            return
        if shared_handle.get("type") != "dask_shared_token":
            raise ValueError(
                "DaskEngine: unsupported shared handle type, expecting 'dask_shared_token'"
            )
        if len(shared_memory):
            client = self._client
            assert client is not None
            client.cancel(shared_memory)
        shared_memory.clear()

    def __del__(self) -> None:
        if hasattr(self, "_client") and self._client is not None:
            self._client.__del__()
        if hasattr(self, "_cluster") and self._cluster is not None:
            self._cluster.__del__()
        super().__del__()

    def _futureize_large_submit_value(self, value: Any, cache: dict[int, Any]) -> Any:
        """Convert large array payloads into dask futures before submission."""
        if isinstance(value, np.ndarray):
            if (
                self.min_submit_array_bytes > 0
                and value.nbytes >= self.min_submit_array_bytes
            ):
                key = id(value)
                if key not in cache:
                    client = self._client
                    assert client is not None
                    cache[key] = client.scatter(
                        np.ascontiguousarray(value),
                        broadcast=False,
                        hash=False,
                    )
                return cache[key]
            return value

        if isinstance(value, Mapping):
            return {
                k: self._futureize_large_submit_value(v, cache)
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._futureize_large_submit_value(v, cache) for v in value]
        if isinstance(value, tuple):
            return tuple(self._futureize_large_submit_value(v, cache) for v in value)
        return value

    def _prepare_submit_payload(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Prepare submit payload by futureizing large arrays in Data containers."""
        if self.min_submit_array_bytes <= 0:
            return args, kwargs

        cache: dict[int, Any] = {}

        def _prepare_single(value: Any) -> Any:
            # Keep typed data containers intact; futureize only their extra_data.
            if hasattr(value, "dims") and hasattr(value, "extra_data"):
                value.extra_data = self._futureize_large_submit_value(
                    value.extra_data, cache
                )
                return value

            if isinstance(value, MutableMapping):
                for k in list(value.keys()):
                    value[k] = self._futureize_large_submit_value(value[k], cache)
            if hasattr(value, "extra_data"):
                value.extra_data = self._futureize_large_submit_value(
                    value.extra_data, cache
                )
            return value

        args = tuple(_prepare_single(a) for a in args)
        kwargs = {k: _prepare_single(v) for k, v in kwargs.items()}
        return args, kwargs

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
        args, kwargs = self._prepare_submit_payload(args, kwargs)
        client = self._client
        assert client is not None
        return client.submit(f, *args, **kwargs)

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

    def new_runner(self) -> DaskProcessRunner:
        """Creates a dask-aware runner for ProcessEngine chunk execution."""
        return DaskProcessRunner()


class SlurmClusterEngine(LocalClusterEngine):
    """
    The dask engine for foxes calculations on a SLURM cluster.
    """

    def __enter__(self) -> LocalClusterEngine:
        distributed_mod = _get_distributed()
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
        self._client = distributed_mod.Client(
            self._cluster, **self.client_pars
        ).__enter__()

        cluster = self._cluster
        client = self._client
        assert cluster is not None
        assert client is not None
        self.print(cluster)
        self.print(f"Dashboard: {client.dashboard_link}\n")
        print(cluster.job_script())

        return LocalClusterEngine.__enter__(self)
