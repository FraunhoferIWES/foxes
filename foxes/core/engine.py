from __future__ import annotations

import os
import numpy as np
from abc import ABC, abstractmethod
from xarray import Dataset
from typing import TYPE_CHECKING, Any, Callable, cast

from foxes.config import config
from foxes.utils import new_instance
import foxes.constants as FC

from .data import MData, FData, TData

if TYPE_CHECKING:
    from .algorithm import Algorithm
    from .data_calc_model import DataCalcModel

__global_engine_data__: dict[str, Engine | None] = dict(engine=None)


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
        """Rebuild chunk-local farm data from previous farm results when available."""
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
        fdata = cast(
            FData,
            FData.from_dataset(
                prev_farm_results,
                mdata=mdata,
                s_states=s_states,
                callback=None,
                states_i0=i0,
                n_states=mdata.n_states,
                n_turbines=algo.n_turbines,
                copy=True,
            ),
        )
        return fdata, True

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the chunk calculation"""
        pass


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

    def __exit__(self, *exit_args: Any) -> None:
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
        pass

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
        pass

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
        pass

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
        # Preserve deterministic output order on every eager backend.
        return [self.await_result(future) for future in futures]

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
        pass

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

    @abstractmethod
    def run_calculation(
        self,
        algo: Algorithm,
        model: DataCalcModel,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        *,
        progress_label: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run the model calculation.

        Parameters
        ----------
        algo
            The algorithm object.
        model
            The model whose ``calculate`` method should be executed.
        model_data
            The initial model data.
        farm_data
            The initial farm data.
        point_data
            The initial point data.
        progress_label
            Operation label for progress displays. ``None`` uses the model class.
        kwargs
            Additional engine-specific calculation options.

        Returns
        -------
        results
            The model results.

        """
        n_states = algo.n_states
        if model_data is not None and FC.STATE in model_data.sizes:
            n_states = model_data.sizes[FC.STATE]
        elif farm_data is not None and FC.STATE in farm_data.sizes:
            n_states = farm_data.sizes[FC.STATE]
        elif point_data is not None and FC.STATE in point_data.sizes:
            n_states = point_data.sizes[FC.STATE]
        if point_data is None:
            self.print(
                f"{self.name}: Calculating {n_states} states for {algo.n_turbines} turbines"
            )
        else:
            self.print(
                f"{self.name}: Calculating data at {point_data.sizes[FC.TARGET]} points for {n_states} states"
            )
        if not model.initialized:
            raise ValueError(f"Model '{model.name}' not initialized")

    @abstractmethod
    def new_runner(self) -> EngineRunner:
        """
        Create a new engine runner for this engine.

        Returns
        -------
        runner
            The engine runner.

        """
        pass

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
