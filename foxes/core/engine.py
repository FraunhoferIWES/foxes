from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.utils import Dict, new_instance

from .data import FData, MData, TData

if TYPE_CHECKING:
    from .algorithm import Algorithm
    from .data_calc_model import DataCalcModel
    from .output import Output

__global_engine_data__: dict[str, Engine | None] = dict(engine=None)


@dataclass
class _Calculation:
    """Per-call caches, output mappings and chunk counts for the lifecycle."""

    chunk_store: dict[Any, Any]
    new_chunk_store: dict[Any, Any] = field(default_factory=dict)
    chunk_results: Dict[str, Dict[tuple[int, int], Any]] = field(
        default_factory=lambda: Dict(_name="chunk_results")
    )
    output_results: Dict[str, Any] = field(
        default_factory=lambda: Dict(_name="output_results")
    )
    n_chunks_states: int = 0
    n_chunks_targets: int = 0


def _expand_farm_fields(data: dict[str, Any], dims: dict[str, Any]) -> None:
    """Broadcast reduced farm fields to the chunk's full farm axes.

    Parameters
    ----------
    data
        Chunk-local dataset values, including ``FC.STATE`` and ``FC.TURBINE``
        coordinates. Reduced fields are replaced in place by read-only views;
        coordinate fields, ``FC.TNAME`` and ``FV.WEIGHT`` are left unchanged.
    dims
        Variable-to-dimension mapping, updated in place to
        ``(FC.STATE, FC.TURBINE)`` for scalar, state-only and turbine-only
        physical fields. Other dimensions are preserved.
    """
    # Farm containers require full state/turbine axes, unlike datasets.
    shape = (len(data[FC.STATE]), len(data[FC.TURBINE]))
    for variable, axes in list(dims.items()):
        if variable in (FC.STATE, FC.TURBINE, FC.TNAME, FV.WEIGHT):
            continue
        if axes == (FC.TURBINE,):
            data[variable] = np.broadcast_to(data[variable][None, :], shape)
        elif axes == (FC.STATE,):
            data[variable] = np.broadcast_to(data[variable][:, None], shape)
        elif axes == ():
            data[variable] = np.broadcast_to(data[variable], shape)
        else:
            continue
        dims[variable] = (FC.STATE, FC.TURBINE)


class EngineRunner(ABC):
    """
    Helper class for running calculations in engines
    """

    def _apply_prev_farm_results(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
    ) -> tuple[FData, bool]:
        """Rebuild chunk-local farm data from available previous results.

        Parameters
        ----------
        algo
            Calculation algorithm, queried for previous farm results when it
            provides ``prev_farm_results``.
        mdata
            Model chunk supplying state offsets and fallback previous results
            through ``FC.PREV_FARM_RESULTS`` in ``extra_data``.
        fdata
            Existing farm chunk, returned unchanged if previous results are
            absent, empty or do not cover the current state interval.

        Returns
        -------
        farm_data
            The original container or a newly constructed chunk in the previous
            results' turbine order. Reduced physical fields are broadcast to
            ``(FC.STATE, FC.TURBINE)``; previous results are not mutated.
        applied
            Whether previous results replaced ``fdata``.
        """
        # Prefer algorithm-specific history over model-data fallback storage.
        prev_farm_results = (
            algo.prev_farm_results(mdata=mdata)
            if hasattr(algo, "prev_farm_results")
            else mdata.extra_data.get(FC.PREV_FARM_RESULTS, None)
        )
        if prev_farm_results is None:
            return fdata, False
        if isinstance(prev_farm_results, Dataset):
            if len(prev_farm_results.data_vars) == 0:
                return fdata, False
            prev_n_states = prev_farm_results.sizes.get(FC.STATE, 0)
            if prev_n_states == 0:
                return fdata, False

        # Reuse an already chunk-sized dataset or slice full-state history.
        i0 = mdata.states_i0(counter=True)
        if i0 is None or mdata.n_states is None:
            return fdata, False
        i1 = i0 + mdata.n_states
        s_states = np.s_[i0:i1]
        if isinstance(prev_farm_results, Dataset):
            prev_n_states = prev_farm_results.sizes.get(FC.STATE, 0)
            if prev_n_states == mdata.n_states:
                s_states = np.s_[:]
            elif prev_n_states < i1:
                return fdata, False

        # Copy prior values before adapting their reduced dataset dimensions.
        fdata = cast(
            FData,
            FData.from_dataset(
                prev_farm_results,
                mdata=mdata,
                s_states=s_states,
                callback=_expand_farm_fields,
                states_i0=i0,
                n_states=mdata.n_states,
                n_turbines=algo.n_turbines,
                copy=True,
            ),
        )
        return fdata, True

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run one engine-specific chunk calculation.

        Parameters
        ----------
        args
            Positional inputs required by the concrete engine runner.
        kwargs
            Keyword inputs required by the concrete engine runner.

        Returns
        -------
        chunk_results
            Calculated chunk payload in the concrete engine's submission format.
        """


class Engine(ABC):
    """
    Abstract base class for foxes calculation engines.

    Notes
    -----
    Use engines via the context manager protocol:
    >>> engine = Engine.new(...)
    >>> with engine:
    >>>     ...
    """

    def __init__(
        self,
        chunk_size_states: int | None = None,
        chunk_size_points: int | None = None,
        n_procs: int | None = None,
        progress_bar: bool | None = True,
        verbosity: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        chunk_size_states
            The size of a states chunk.
        chunk_size_points
            The size of a points chunk.
        n_procs
            The number of processes to be used, or ``None`` for automatic
            selection.
        progress_bar
            True selects a terminal/notebook progress bar, falling back to
            bounded newline-only progress snapshots in redirected logs.
            False always uses snapshots; ``None`` disables progress.
            Verbosity zero suppresses progress in every mode.
        verbosity
            The verbosity level, where ``0`` is silent.
        """
        self.chunk_size_states = chunk_size_states
        self.chunk_size_points = chunk_size_points
        self.progress_bar = progress_bar
        self.verbosity = verbosity

        self._n_procs = n_procs if n_procs is not None else os.cpu_count() or 1
        self._n_workers = max(self._n_procs - 1, 1)

        self.__name = type(self).__name__
        self.__entered = False
        self.__running_chunk_calc = False

    def __repr__(self) -> str:
        s = f"n_procs={self.n_procs}, chunk_size_states={self.chunk_size_states}, chunk_size_points={self.chunk_size_points}"
        return f"{self.name}({s})"

    def __enter__(self) -> Engine:
        if self.__entered:
            raise ValueError(
                f"Engine '{self.name}': Enter called for already entered engine"
            )
        if has_engine():
            raise ValueError(
                f"Cannot enter engine '{self.name}', since engine already set to '{type(get_engine()).__name__}'"
            )
        self.__entered = True
        __global_engine_data__["engine"] = self
        return self

    def __exit__(self, *exit_args: object) -> None:
        """Finalize the engine and release its active context.

        Parameters
        ----------
        exit_args
            Exception type, exception value and traceback supplied by the
            context-manager protocol. Exceptions are not suppressed.

        Raises
        ------
        ValueError
            If the engine has not entered its context.
        """
        if not self.__entered:
            raise ValueError(
                f"Engine '{self.name}': Exit called for not entered engine"
            )
        self.__entered = False
        __global_engine_data__["engine"] = None

    def __del__(self) -> None:
        if self.__entered:
            __global_engine_data__["engine"] = None

    @property
    def name(self) -> str:
        """
        Return the engine name.

        Returns
        -------
        nme
            The engine name.

        """
        return self.__name

    @property
    def n_procs(self) -> int:
        """
        Return the number of processes.

        Returns
        -------
        n_procs
            The number of processes.

        """
        return self._n_procs

    @property
    def n_workers(self) -> int:
        """
        Return the number of worker processes.

        Returns
        -------
        n_workers
            The number of worker processes.

        """
        return self._n_workers

    @property
    def has_progress_bar(self) -> bool:
        """
        Return whether a progress bar is active.

        Returns
        -------
        has_pbar
            ``True`` if a progress bar is active.

        """
        return self.progress_bar is not None and self.progress_bar

    @property
    def prints_progress(self) -> bool:
        """
        Return whether progress printing is active.

        Returns
        -------
        has_pbar
            ``True`` if progress printing is active.

        """
        return self.progress_bar is not None and not self.progress_bar

    @property
    def entered(self) -> bool:
        """
        Return whether this engine has been entered.

        Returns
        -------
        flag
            ``True`` if the engine has been entered.

        """
        return self.__entered

    @property
    def running_chunk_calc(self) -> bool:
        """
        Return whether a chunk calculation is running.

        Returns
        -------
        flag
            ``True`` if a chunk calculation is running.

        """
        return self.__running_chunk_calc

    def print(self, *args: Any, level: int = 1, **kwargs: Any) -> None:
        """Print output based on the configured verbosity."""
        if self.verbosity >= level:
            print(*args, **kwargs)

    @abstractmethod
    def submit(self, f: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Submit a job to a worker and return the future.

        Parameters
        ----------
        f
            The function to be submitted.
        args
            Positional arguments for the function.
        kwargs
            Keyword arguments for the function.

        Returns
        -------
        future
            The future object.

        """

    @abstractmethod
    def future_is_done(self, future: Any) -> bool:
        """
        Check whether a future is done.

        Parameters
        ----------
        future
            The future.

        Returns
        -------
        is_done
            ``True`` if the future is done.

        """

    @abstractmethod
    def await_result(self, future: Any) -> Any:
        """
        Wait for and return the result of a future.

        Parameters
        ----------
        future
            The future.

        Returns
        -------
        result
            The calculation result.

        """

    def await_results(self, futures: list[Any]) -> list[Any]:
        """Resolve a batch of tasks in submission order.

        Parameters
        ----------
        futures
            Submitted tasks. Lazy backends may schedule this batch together.

        Returns
        -------
        results
            Task results in the same order, with worker exceptions propagated.
        """
        return list(self.iter_results(futures))

    def iter_results(self, futures: list[Any]) -> Iterator[Any]:
        """Yield submitted task results incrementally in submission order.

        Parameters
        ----------
        futures
            Submitted tasks. Lazy backends must schedule their graphs together.

        Yields
        ------
        result
            The next task result, with worker exceptions propagated before
            requesting subsequent results.
        """
        for future in futures:
            yield self.await_result(future)

    @abstractmethod
    def map(
        self,
        func: Callable[..., Any],
        inputs: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Run a function on a list of inputs.

        Parameters
        ----------
        func
            The function to call for each input.
        inputs
            The input data list.
        args
            Additional positional arguments for ``func``.
        kwargs
            Additional keyword arguments for ``func``.

        Returns
        -------
        results
            The result list.

        """

    @property
    def loop_dims(self) -> list[str]:
        """
        Return the loop dimensions, including chunking when applicable.

        Returns
        -------
        dims
            The loop dimensions, possibly chunked.

        """
        if self.chunk_size_states is None and self.chunk_size_points is None:
            return []
        elif self.chunk_size_states is None:
            return [FC.TARGET]
        elif self.chunk_size_points is None:
            return [FC.STATE]
        else:
            return [FC.STATE, FC.TARGET]

    def select_subsets(
        self,
        *datasets: Any,
        sel: dict[str, Any] | None = None,
        default_n_states: int | None = None,
    ) -> tuple[list[Any], int | None]:
        """
        Take subsets of datasets.

        Parameters
        ----------
        datasets
            The ordered xarray dataset or data array objects.
        sel
            Coordinate-label selectors to apply with ``sel``.
        default_n_states
            The fallback number of states if no dataset has a state dimension.

        Returns
        -------
        subsets
            The subsets of the input data.
        n_states
            The number of states after subset selection, or the fallback value.

        """
        subsets: list[Any] = list(datasets)

        if sel is not None:
            new_datasets: list[Any] = []
            for data in subsets:
                if data is not None:
                    missing = [
                        c for c in sel if c in data.dims and c not in data.coords
                    ]
                    if missing:
                        raise ValueError(
                            f"Cannot select dimensions without coordinates: {missing}"
                        )
                    s = {c: u for c, u in sel.items() if c in data.coords}
                    non_unique = [c for c in s if not data.get_index(c).is_unique]
                    if non_unique:
                        raise ValueError(
                            f"Cannot select dimensions with duplicate labels: {non_unique}"
                        )
                    new_datasets.append(data.sel(s) if len(s) else data)
                else:
                    new_datasets.append(data)
            subsets = new_datasets

        n_states = default_n_states
        for data in subsets:
            if data is not None and FC.STATE in data.sizes:
                n_states = data.sizes[FC.STATE]
                break

        return subsets, n_states

    def calc_chunk_sizes(
        self,
        n_states: int,
        n_targets: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the sizes of the state and target chunks.

        Parameters
        ----------
        n_states
            The number of states.
        n_targets
            The number of point targets.

        Returns
        -------
        chunk_sizes_states
            The sizes of all state chunks, with shape ``(n_chunks_states,)``.
        chunk_sizes_targets
            The sizes of all target chunks, with shape ``(n_chunks_targets,)``.

        """
        # determine states chunks:
        if self.chunk_size_states is None:
            n_chunks_states = min(self.n_workers, n_states)
            chunk_size_states = max(int(n_states / self.n_workers), 1)
        else:
            chunk_size_states = min(n_states, self.chunk_size_states)
            n_chunks_states = max(int(n_states / chunk_size_states), 1)
            if int(n_states / n_chunks_states) > chunk_size_states:
                n_chunks_states += 1
                chunk_size_states = int(n_states / n_chunks_states)

        # determine points chunks:
        chunk_sizes_targets: np.ndarray = np.asarray(
            [n_targets], dtype=config.dtype_int
        )
        if n_targets > 1:
            if self.chunk_size_points is None:
                if n_targets < max(n_states, 1000):
                    chunk_size_targets = n_targets
                    n_chunks_targets = 1
                else:
                    n_chunks_targets = min(self.n_workers, n_targets)
                    chunk_size_targets = max(int(n_targets / self.n_workers), 1)
                    if self.chunk_size_states is None and n_chunks_states > 1:
                        while chunk_size_states * chunk_size_targets > n_targets:
                            n_chunks_states += 1
                            chunk_size_states = int(n_states / n_chunks_states)
            else:
                chunk_size_targets = min(n_targets, self.chunk_size_points)
                n_chunks_targets = max(int(n_targets / chunk_size_targets), 1)
            if int(n_targets / n_chunks_targets) > chunk_size_targets:
                n_chunks_targets += 1
                chunk_size_targets = int(n_targets / n_chunks_targets)
            chunk_sizes_targets = np.full(
                n_chunks_targets, chunk_size_targets, dtype=config.dtype_int
            )
            extra = n_targets - n_chunks_targets * chunk_size_targets
            if extra > 0:
                chunk_sizes_targets[-extra:] += 1

            s = np.sum(chunk_sizes_targets)
            assert s == n_targets, (
                f"Targets count mismatch: Expecting {n_targets}, chunks sum is {s}. Chunks: {[int(c) for c in chunk_sizes_targets]}"
            )

        chunk_sizes_states = np.full(n_chunks_states, chunk_size_states)
        extra = n_states - n_chunks_states * chunk_size_states
        if extra > 0:
            chunk_sizes_states[-extra:] += 1

        s = np.sum(chunk_sizes_states)
        assert s == n_states, (
            f"States count mismatch: Expecting {n_states}, chunks sum is {s}. Chunks: {[int(c) for c in chunk_sizes_states]}"
        )

        return chunk_sizes_states, chunk_sizes_targets

    def get_chunk_input_data(
        self,
        algo: Algorithm,
        model_data: Dataset,
        farm_data: Dataset | None,
        point_data: Dataset | None,
        states_i0_i1: tuple[int, int],
        targets_i0_i1: tuple[int, int],
        chunki_states: int,
        chunki_points: int,
        n_chunks_states: int,
        n_chunks_points: int,
    ) -> tuple[MData, FData] | tuple[MData, FData, TData]:
        """
        Extract the data for a single chunk calculation.

        Parameters
        ----------
        algo
            The algorithm object.
        model_data
            The initial model data.
        farm_data
            The initial farm data.
        point_data
            The initial point data.
        states_i0_i1
            The start and end indices of the state slice.
        targets_i0_i1
            The start and end indices of the target slice.
        chunki_states
            The index of the states chunk.
        chunki_points
            The index of the points chunk.
        n_chunks_states
            The number of state chunks.
        n_chunks_points
            The number of point chunks.

        Returns
        -------
        data
            The input data for the chunk calculation, either ``(mdata, fdata)``
            or ``(mdata, fdata, tdata)``.

        """
        # prepare:
        i0_states, i1_states = states_i0_i1
        i0_targets, i1_targets = targets_i0_i1
        s_states = np.s_[i0_states:i1_states]
        s_targets = np.s_[i0_targets:i1_targets]
        n_states = i1_states - i0_states

        # special case for sequential algo:
        if hasattr(algo, "states_i0"):
            i0_states = algo.states_i0(counter=True)

        # create mdata:
        mdata = cast(
            MData,
            MData.from_dataset(
                model_data,
                s_states=s_states,
                loop_dims=[FC.STATE],
                states_i0=i0_states,
                copy=True,
                chunki_states=chunki_states,
                chunki_points=chunki_points,
                n_chunks_states=n_chunks_states,
                n_chunks_points=n_chunks_points,
                n_states=n_states,
                n_turbines=algo.n_turbines,
            ),
        )

        # create fdata:
        if farm_data is not None:
            fdata = cast(
                FData,
                FData.from_dataset(
                    farm_data,
                    mdata=mdata,
                    s_states=s_states,
                    callback=None,
                    states_i0=i0_states,
                    n_states=n_states,
                    n_turbines=algo.n_turbines,
                    copy=True,
                ),
            )
        else:
            fdata = cast(
                FData,
                FData.from_data(
                    base_data=mdata,
                    states_i0=i0_states,
                ),
            )

        # create tdata:
        tdata = (
            cast(
                TData,
                TData.from_dataset(
                    point_data,
                    mdata=mdata,
                    s_states=s_states,
                    s_targets=s_targets,
                    callback=None,
                    states_i0=i0_states,
                    n_states=n_states,
                    n_turbines=algo.n_turbines,
                    copy=True,
                ),
            )
            if point_data is not None
            else None
        )

        return (mdata, fdata) if tdata is None else (mdata, fdata, tdata)

    def get_start_calc_message(
        self,
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> str:
        """Helper function for start calculation message"""
        msg = f"{self.name}: Starting calculation using "
        if self.n_workers > 1:
            msg += f"{self.n_workers} workers"
        else:
            msg += "a single worker"
        if n_chunks_states > 1 or n_chunks_targets > 1:
            msg += f", for {n_chunks_states} states chunks"
            if n_chunks_targets > 1:
                msg += f" and {n_chunks_targets} targets chunks"
        msg += "."
        return msg

    def run_calculation(
        self,
        algo: Algorithm,
        model: DataCalcModel | None,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
        output: Output | None = None,
        chunk_store: dict[Any, Any] | None = None,
        sel: dict[str, Any] | None = None,
        iterative: bool = False,
        *,
        progress_label: str | None = None,
        **calc_pars: Any,
    ) -> Dict[str, Any]:
        """
        Run backend chunks within the common output and cache lifecycle.

        Parameters
        ----------
        algo
            The algorithm object.
        model
            The model whose ``calculate`` method should be executed. ``None``
            runs only initial results and parent-side output finalization,
            without dispatching scientific chunks or allocating worker resources.
        model_data
            The initial model data.
        farm_data
            The initial farm data.
        point_data
            The initial point data.
        extra_data
            Non-array model inputs forwarded to the backend.
        output
            Initialized, non-running output model, or ``None`` to omit output
            collection. Remains running through parent-side result finalization.
        chunk_store
            Cached scientific data, reattached after backend cleanup even on
            failure. ``None`` starts empty.
        sel
            Coordinate subsets passed to backend selection and output hooks.
        iterative
            Whether this calculation is an iterative algorithm pass.
        progress_label
            Operation label for progress displays. ``None`` uses the model class.
        calc_pars
            Additional parameters forwarded to the scientific model.

        Returns
        -------
        output_results
            Result-name keyed output mapping, empty without a selected output.

        Notes
        -----
        Worker cache updates are merged only on success. Reattachment is not
        rollback: in-place mutations to shared caches can survive failures.
        Backend cleanup precedes output restoration and cache reattachment.
        """
        # Reject invalid outputs before taking ownership of their lifecycle.
        if output is not None:
            if not output.initialized:
                raise ValueError(
                    f"Engine '{self.name}': Output model '{output.name}' is not initialized"
                )
            if output.running:
                raise ValueError(
                    f"Engine '{self.name}': Output model '{output.name}' is already running"
                )
            results = output.get_initial_output_results(
                algo,
                model_data=model_data,
                farm_data=farm_data,
                point_data=point_data,
                extra_data=extra_data,
            )
        else:
            results = Dict[str, Any](_name="results")
        calculation = self._create_calculation(
            {} if chunk_store is None else chunk_store
        )
        output_data_stash: dict[str, dict[str, Any]] = {}
        failed = True

        # Restore outputs and caches even if backend preparation or cleanup fails.
        try:
            try:
                algo.reset_chunk_store()
                if output is not None:
                    output.set_running(
                        algo, output_data_stash, sel=sel, verbosity=self.verbosity - 2
                    )
                if model is not None:
                    results = self._run_calculation(
                        calculation,
                        algo,
                        model,
                        results,
                        model_data,
                        farm_data,
                        point_data,
                        extra_data=extra_data,
                        output=output,
                        sel=sel,
                        iterative=iterative,
                        progress_label=progress_label,
                        **calc_pars,
                    )
                if output is not None:
                    output.finalize_results(
                        algo=algo,
                        n_chunks_states=calculation.n_chunks_states,
                        n_chunks_targets=calculation.n_chunks_targets,
                        output_results=results,
                    )
                failed = False
            finally:
                self._cleanup_calculation(calculation, failed)

            calculation.chunk_store.update(calculation.new_chunk_store)
            return results
        finally:
            try:
                if output is not None and output.running:
                    output.unset_running(
                        algo, output_data_stash, sel=sel, verbosity=self.verbosity - 2
                    )
            finally:
                algo.reset_chunk_store(calculation.chunk_store)

    @abstractmethod
    def _create_calculation(self, chunk_store: dict[Any, Any]) -> _Calculation:
        """Create per-call state without allocating backend resources.

        Parameters
        ----------
        chunk_store
            Scientific caches retained by the common lifecycle.

        Returns
        -------
        calculation
            Backend state, safe to clean up before preparation has completed.
        """

    @abstractmethod
    def _run_calculation(
        self,
        calculation: _Calculation,
        algo: Algorithm,
        model: DataCalcModel,
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
        """Prepare and execute chunks without finalizing the output lifecycle.

        Parameters
        ----------
        calculation
            Per-call state. Record resources as they are acquired, chunk counts
            and collected cache updates here for finalization and cleanup.
        algo
            Running calculation algorithm.
        model
            Initialized scientific model.
        results
            Initial output results mapping, updated in place and returned.
        model_data
            Initial model dataset.
        farm_data
            Initial farm dataset or previous farm results.
        point_data
            Initial target dataset.
        extra_data
            Non-array model inputs.
        output
            Running output model, or ``None`` to omit output collection.
        sel
            Coordinate subsets.
        iterative
            Whether this is an iterative pass.
        progress_label
            Operation label for progress displays.
        calc_pars
            Scientific calculation parameters.

        Returns
        -------
        results
            Collected result-name mapping awaiting parent-side finalization.
        """

    @abstractmethod
    def _cleanup_calculation(self, calculation: _Calculation, failed: bool) -> None:
        """Finish workers and release resources, including after partial setup.

        Parameters
        ----------
        calculation
            State populated by backend execution, possibly only partially.
        failed
            Whether setup, execution or output finalization failed.
        """

    def _collect_results(
        self,
        calculation: _Calculation,
        algo: Algorithm,
        output: Output | None,
        futures: dict[tuple[int, int], Any],
        on_chunk: Callable[[], None],
    ) -> None:
        """Resolve an ordered batch and merge chunk outputs and cache updates.

        Parameters
        ----------
        calculation
            Per-call result mappings, cache updates and total chunk counts.
        algo
            Running calculation algorithm passed to output hooks.
        output
            Running output model, or ``None`` to collect only cache updates.
        futures
            State/target keys and pending handles in submission order. Remove
            each resolved entry before consuming its payload; leave pending
            entries available to backend cleanup on failure.
        on_chunk
            Callback invoked after a chunk and its output updates succeed.
        """
        # Consume each ordered result without waiting for the rest of the batch.
        keys = list(futures)
        batch = self.iter_results(list(futures.values()))
        for key, (results, caches) in zip(keys, batch):
            del futures[key]
            if results is not None:
                for name, result in results.items():
                    if name not in calculation.chunk_results:
                        calculation.chunk_results[name] = Dict({key: result})
                    elif key not in calculation.chunk_results[name]:
                        calculation.chunk_results[name][key] = result
                    else:
                        calculation.chunk_results[name][key].update(result)
            for cache_key, entry in caches.items():
                calculation.new_chunk_store.setdefault(cache_key, {}).update(entry)

            # Outputs consume their pending keys before progress records success.
            if output is not None:
                output.update_results(
                    algo,
                    n_chunks_states=calculation.n_chunks_states,
                    n_chunks_targets=calculation.n_chunks_targets,
                    new_chunk_results=calculation.chunk_results,
                    output_results=calculation.output_results,
                )
            on_chunk()

    def _prepare_calculation_inputs(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        model_data: Dataset | None,
        farm_data: Dataset | None,
        point_data: Dataset | None,
        extra_data: dict[str, Any],
        sel: dict[str, Any] | None,
    ) -> tuple[Dataset, Dataset, Dataset | None, int]:
        """Select and validate scientific inputs independently of dispatch.

        Parameters
        ----------
        algo
            Algorithm supplying the fallback state count.
        model
            Scientific model whose initialization is checked.
        model_data
            Required model dataset.
        farm_data
            Farm fields, or previous farm results when no target data is given.
        point_data
            Target dataset, or ``None`` for a farm calculation.
        extra_data
            Non-array inputs, updated in place with previous farm results.
        sel
            Coordinate-label subsets applied to all supplied datasets.

        Returns
        -------
        model_data
            Selected model dataset.
        farm_data
            Selected farm dataset, empty when absent or moved to ``extra_data``.
        point_data
            Selected target dataset, or ``None``.
        n_states
            State count after selection.

        Raises
        ------
        ValueError
            If model data or the state count is missing, selection is invalid,
            or the scientific model is not initialized.
        """
        # Select before interpreting farm inputs or constructing empty containers.
        (model_data, farm_data, point_data), n_states = self.select_subsets(
            model_data, farm_data, point_data, sel=sel, default_n_states=algo.n_states
        )
        if model_data is None:
            raise ValueError(f"Engine '{self.name}': Missing model_data")
        if n_states is None:
            raise ValueError(f"Engine '{self.name}': Could not determine n_states")
        if farm_data is not None and point_data is None:
            extra_data[FC.PREV_FARM_RESULTS] = farm_data
            farm_data = None
        self._validate_calculation(algo, model, model_data, farm_data, point_data)
        if farm_data is None:
            farm_data = Dataset()
        return model_data, farm_data, point_data, n_states

    def _get_execution_description(self) -> str:
        """Describe the execution resources used by calculation progress.

        Returns
        -------
        description
            Worker count and its singular or plural label.
        """
        return f"{self.n_workers} worker{'s' if self.n_workers != 1 else ''}"

    def _validate_calculation(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        model_data: Dataset | None,
        farm_data: Dataset | None,
        point_data: Dataset | None,
    ) -> None:
        """Validate initialization and report the calculation domain and workers."""
        n_states = algo.n_states
        if model_data is not None and FC.STATE in model_data.sizes:
            n_states = model_data.sizes[FC.STATE]
        elif farm_data is not None and FC.STATE in farm_data.sizes:
            n_states = farm_data.sizes[FC.STATE]
        elif point_data is not None and FC.STATE in point_data.sizes:
            n_states = point_data.sizes[FC.STATE]
        execution = self._get_execution_description()
        if point_data is None:
            self.print(
                f"{self.name}: Calculating {n_states} states for {algo.n_turbines} turbines using {execution}"
            )
        else:
            self.print(
                f"{self.name}: Calculating data at {point_data.sizes[FC.TARGET]} points for {n_states} states using {execution}"
            )
        if not model.initialized:
            raise ValueError(f"Model '{model.name}' not initialized")

    @classmethod
    def new(cls, engine_type: str | None, *args: Any, **kwargs: Any) -> Engine:
        """
        Create an engine instance at runtime.

        Parameters
        ----------
        engine_type
            The selected derived class name.
        args
            Additional positional arguments for the constructor.
        kwargs
            Additional keyword arguments for the constructor.

        """

        if engine_type is None:
            engine_type = "default"

        engine_type = dict(
            default="DefaultEngine",
            threads="ThreadsEngine",
            process="ProcessEngine",
            dask="DaskEngine",
            multiprocess="MultiprocessEngine",
            local_cluster="LocalClusterEngine",
            slurm_cluster="SlurmClusterEngine",
            mpi="MPIEngine",
            ray="RayEngine",
            numpy="NumpyEngine",
            single="SingleChunkEngine",
        ).get(engine_type, engine_type)

        return cast(Engine, new_instance(cls, engine_type, *args, **kwargs))


def get_engine(error: bool = True) -> Engine | None:
    """
    Return the global calculation engine.

    Parameters
    ----------
    error
        Whether to raise a ``ValueError`` if no engine is set.

    Returns
    -------
    engine
        The foxes calculation engine.


    """
    engine = __global_engine_data__.get("engine", None)
    if engine is None and error:
        raise ValueError("No engine has been set.")
    return engine


def has_engine() -> bool:
    """
    Return whether an engine has been set.

    Returns
    -------
    flag
        ``True`` if an engine has been set.


    """
    return __global_engine_data__.get("engine", None) is not None


def run_with_engine(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Run a function within the active engine context.

    Parameters
    ----------
    func
        The function to run.
    args
        Arguments for the function.
    kwargs
        Keyword arguments for the function.

    Returns
    -------
    result
        The function result.

    """
    if has_engine():
        results = func(*args, **kwargs)
    else:
        with Engine.new("default"):
            results = func(*args, **kwargs)
    return results


def map_with_engine(*args: Any, **kwargs: Any) -> Any:
    """
    Map a function via the active engine.

    Parameters
    ----------
    args
        Arguments for the ``Engine.map`` function.
    kwargs
        Keyword arguments for the ``Engine.map`` function.

    Returns
    -------
    result
        The function result.

    """
    if has_engine():
        engine = get_engine()
        assert engine is not None
        results = engine.map(*args, **kwargs)
    else:
        with Engine.new("default") as e:
            results = e.map(*args, **kwargs)
    return results


def launch_parallel_calc(
    self: Any,
    *args: Any,
    progress_label: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Launch a parallel calculation using the active engine.

    Parameters
    ----------
    args
        Additional parameters for running the calculation.
    progress_label
        Upstream operation label, forwarded unchanged to the active engine.
        ``None`` retains the engine's model-class label.
    kwargs
        Additional keyword arguments for running the calculation.

    Returns
    -------
    results
        The calculation results.

    """
    if has_engine():
        engine = get_engine()
        assert engine is not None
        results = engine.run_calculation(
            self, *args, progress_label=progress_label, **kwargs
        )
    else:
        with Engine.new("default") as e:
            results = e.run_calculation(
                self, *args, progress_label=progress_label, **kwargs
            )
    return results
