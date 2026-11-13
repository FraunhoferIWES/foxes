from __future__ import annotations

import sys
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
from foxes.config import config
from foxes.core import Engine
from foxes.core.engine import _Calculation
from foxes.utils import Dict

from .progress import CalculationProgress

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, EngineRunner, MData, Output


@dataclass
class _PoolCalculation(_Calculation):
    """Pool resources retained until common lifecycle cleanup."""

    futures: dict[tuple[int, int], Any] = field(default_factory=dict)
    shared_memory: list[Any] = field(default_factory=list)
    shared_handle: Any = None
    progress: CalculationProgress | None = None


def _run_map(func: Any, inputs: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Helper function for running map func on proc"""
    return [func(x, *args, **kwargs) for x in inputs]


class PoolEngine(Engine):
    """Abstract engine for pool-based chunk execution.

    Configure the worker pool, shared inputs and chunk-store ownership.
    """

    def __init__(
        self,
        *args: Any,
        share_cstore: bool = False,
        pool_args: dict[str, Any] | None = None,
        supports_shared_data: bool = True,
        min_shared_array_bytes: int = 65536,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Positional parameters for ``Engine``.
        share_cstore
            Share the engine-managed chunk store between chunks.
        pool_args
            Keyword arguments for the pool constructor. ``None`` uses an empty
            mapping.
        supports_shared_data
            Enable backend-supported shared inputs for chunk calculations.
        min_shared_array_bytes
            Minimum array size in bytes for placing model data into process
            shared storage. Arrays with ``nbytes`` less than or equal to this
            threshold are transferred inline to workers. Supported top-level
            ``extra_data`` values use their complete payload size.
        kwargs
            Additional keyword parameters for ``Engine``.
        """
        super().__init__(*args, **kwargs)
        self._pool: Any = None
        self.share_cstore = share_cstore
        self.pool_args = {} if pool_args is None else pool_args
        self.supports_shared_data = supports_shared_data
        self.min_shared_array_bytes = int(min_shared_array_bytes)

    @abstractmethod
    def new_runner(self) -> EngineRunner:
        """Create a lightweight worker runner without parent-side pool state.

        Returns
        -------
        runner
            Backend-specific runner whose bound ``run`` method can be submitted
            without serializing the engine or its execution resources.
        """

    @abstractmethod
    def _create_pool(self) -> None:
        """Creates the pool"""
        pass

    @abstractmethod
    def _shutdown_pool(self) -> None:
        """Shuts down the pool"""
        pass

    def _print_shared_data(self, shared_mdata: Any, verbosity: int) -> None:
        """Print diagnostics for data that is actually backed by shared storage."""
        if (
            verbosity > 1
            and shared_mdata is not None
            and (len(shared_mdata) > 0 or len(shared_mdata.extra_data) > 0)
        ):
            print(f"\n\n{type(self).__name__} shared data:\n")
            print(shared_mdata)
            print()

    def __enter__(self) -> PoolEngine:
        # Claim the engine slot before allocating backend resources.
        super().__enter__()
        try:
            self._create_pool()
        except BaseException:
            super().__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exit_args: Any) -> None:
        # Release the global slot even when backend shutdown reports an error.
        try:
            self._shutdown_pool()
        finally:
            super().__exit__(*exit_args)

    def _drain_failed_tasks(self, futures: dict[Any, Any]) -> None:
        """Finish outstanding workers before releasing their shared input storage."""
        # Report secondary failures without replacing the original calculation error.
        for key, future in futures.items():
            try:
                self.await_result(future)
            except Exception as error:
                warnings.warn(
                    f"{self.name}: Pending chunk {key} also failed during cleanup: "
                    f"{type(error).__name__}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def map(
        self,
        func: Any,
        inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Runs a function on a list of files

        Parameters
        ----------
        func
            Function to be called on each file,
            ``func(input, *args, **kwargs) -> data``
        inputs
            The input data list
        args
            Arguments for func
        kwargs
            Keyword arguments for func

        Returns
        -------
        results
            Results for the submitted inputs

        """
        if len(inputs) == 0:
            return []
        elif len(inputs) == 1:
            return [func(inputs[0], *args, **kwargs)]
        else:
            inptl = np.array_split(inputs, min(self.n_workers, len(inputs)))
            futures = []
            for subi in inptl:
                futures.append(self.submit(_run_map, func, subi, *args, **kwargs))
            results = []
            for f in futures:
                results += self.await_result(f)
            return results

    def _create_calculation(self, chunk_store: dict[Any, Any]) -> _PoolCalculation:
        # Shared backends update the supplied cache directly.
        return _PoolCalculation(
            chunk_store=chunk_store,
            new_chunk_store=chunk_store if self.share_cstore else {},
        )

    def _cleanup_calculation(self, calculation: _Calculation, failed: bool) -> None:
        # Workers must finish before shared storage and progress are released.
        assert isinstance(calculation, _PoolCalculation)
        try:
            if failed:
                self._drain_failed_tasks(calculation.futures)
        finally:
            try:
                self.release_shared_memory(
                    calculation.shared_memory, calculation.shared_handle
                )
            finally:
                if calculation.progress is not None:
                    calculation.progress.close(
                        error=failed or sys.exc_info()[0] is not None
                    )

    def _run_calculation(
        self,
        calculation: _Calculation,
        algo: Algorithm,
        model: DataCalcModel | None,
        results: Dict[str, Any],
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        *,
        extra_data: dict[str, Any] | None = None,
        output: Output | None = None,
        sel: dict[str, Any] | None = None,
        iterative: bool = False,
        progress_label: str | None = None,
        **calc_pars: Any,
    ) -> Dict[str, Any]:
        """Prepare pool chunks and collect worker payloads for base finalization."""
        # prepare data:
        assert isinstance(calculation, _PoolCalculation)
        calculation.output_results = results
        chunk_store = calculation.chunk_store
        extra_data = {} if extra_data is None else extra_data
        if self.share_cstore:
            algo.reset_chunk_store(chunk_store)

        # Select inputs before validating and partitioning the calculation.
        model_data, farm_data, point_data, n_states = self._prepare_calculation_inputs(
            algo, model, model_data, farm_data, point_data, extra_data, sel
        )
        out_dims = model.output_coords() if model is not None else ()
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(
            n_states, n_targets
        )
        n_chunks_states = calculation.n_chunks_states = len(chunk_sizes_states)
        n_chunks_targets = calculation.n_chunks_targets = len(chunk_sizes_targets)
        self.print(
            f"{type(self).__name__}: Selecting n_chunks_states = {n_chunks_states}, n_chunks_targets = {n_chunks_targets}",
            level=2,
        )

        # Keep resources available to cleanup even when submission fails.
        futures = calculation.futures
        runner = self.new_runner()
        progress = calculation.progress = CalculationProgress(
            f"{self.name} [{type(model).__name__ if progress_label is None else progress_label}]",
            n_chunks_states * n_chunks_targets,
            self.progress_bar,
            self.verbosity,
        )

        # Submit bounded batches in state/target order.
        i0_states = 0
        for chunki_states in range(n_chunks_states):
            i1_states = i0_states + chunk_sizes_states[chunki_states]
            i0_targets = 0
            for chunki_points in range(n_chunks_targets):
                key = (chunki_states, chunki_points)
                i1_targets = i0_targets + chunk_sizes_targets[chunki_points]
                data = self.get_chunk_input_data(
                    algo=algo,
                    model_data=model_data,
                    farm_data=farm_data,
                    point_data=point_data,
                    states_i0_i1=(i0_states, i1_states),
                    targets_i0_i1=(i0_targets, i1_targets),
                    chunki_states=chunki_states,
                    chunki_points=chunki_points,
                    n_chunks_states=n_chunks_states,
                    n_chunks_points=n_chunks_targets,
                )
                mdata_chunk = data[0]
                fdata_chunk = data[1]
                tdata_chunk = data[2] if len(data) > 2 else None
                mdata_chunk.extra_data.update(extra_data)
                if self.supports_shared_data:
                    if calculation.shared_handle is None and (
                        len(mdata_chunk) > 0 or len(extra_data) > 0
                    ):
                        shared_mdata = mdata_chunk.pop_shared(
                            min_shared_array_bytes=self.min_shared_array_bytes
                        )
                        calculation.shared_handle = self.init_shared_memory(
                            calculation.shared_memory,
                            mdata=mdata_chunk,
                            shared_mdata=shared_mdata,
                            verbosity=max(self.verbosity, algo.verbosity),
                        )
                        del shared_mdata
                    elif calculation.shared_handle is not None:
                        self.prepare_chunk_mdata_for_shared(
                            mdata_chunk, calculation.shared_handle
                        )

                futures[key] = self.submit(
                    runner.run,
                    algo,
                    model,
                    mdata_chunk,
                    fdata_chunk,
                    tdata_chunk,
                    output=output,
                    shared=calculation.shared_handle,
                    chunk_store=chunk_store,
                    chunk_key=key,
                    out_dims=out_dims,
                    utm_zone=config.utm_zone if config.utm_zone_set else None,
                    **calc_pars,
                )
                del data, mdata_chunk, fdata_chunk, tdata_chunk
                if (
                    self.future_is_done(futures[key])
                    or len(futures) >= self.n_workers * 3
                ):
                    self._collect_results(
                        calculation, algo, output, futures, progress.update
                    )
                i0_targets = i1_targets
            i0_states = i1_states

        # Leave result finalization and resource cleanup to the base lifecycle.
        if futures:
            self._collect_results(calculation, algo, output, futures, progress.update)
        progress.finalizing()
        return calculation.output_results

    def init_shared_memory(
        self,
        shared_memory: list[Any],
        mdata: MData,
        shared_mdata: Any,
        verbosity: int = 0,
    ) -> Any:
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
        mdata.recombine_with_shared(shared_mdata)
        return None

    def prepare_chunk_mdata_for_shared(self, mdata: MData, shared_handle: Any) -> None:
        """Hook for engine-specific mdata adjustments before worker submission.

        Parameters
        ----------
        mdata
            The chunk model data that will be sent to workers.
        shared_handle
            The handle that describes shared data for worker recombination.

        """
        pass

    def _prepare_chunk_extra_data_for_shared(
        self, mdata: MData, shared_handle: Any
    ) -> None:
        """Remove extra data that the shared handle restores in workers."""
        for key in shared_handle.get("extra_data_keys", ()):
            mdata.extra_data.pop(key, None)

    def release_shared_memory(
        self, shared_memory: list[Any], shared_handle: Any
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
        pass
