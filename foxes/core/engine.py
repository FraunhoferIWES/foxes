from __future__ import annotations

import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm.autonotebook import tqdm
from xarray import Dataset
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

from foxes.config import config, get_output_path
from foxes.utils import new_instance
from foxes.utils import write_nc as write_nc_file
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

    def _write_chunk_results(
        self,
        algo: Algorithm,
        results: dict[str, np.ndarray],
        write_nc: dict[str, Any] | None,
        out_dims: tuple[str, ...],
        mdata: MData,
    ) -> dict[str, np.ndarray] | None:
        """Helper function for optionally writing chunk results to netCDF file"""
        ret_data = True
        if write_nc is not None and write_nc["split"] == "chunks":
            ret_data = write_nc.get("ret_data", False)
            out_dir = get_output_path(write_nc.get("out_dir", "."))
            base_name = write_nc["base_name"]
            ret_data = write_nc.get("ret_data", False)
            out_dir.mkdir(parents=True, exist_ok=True)

            coords: dict[str, np.ndarray] = {}
            if FC.STATE in out_dims and FC.STATE in mdata:
                coords[FC.STATE] = mdata[FC.STATE]

            dvars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
            for v, d in results.items():
                if (
                    out_dims == (FC.STATE, FC.TURBINE)
                    and d.shape[1] == 1
                    and algo.n_turbines > 1
                ):
                    dvars[v] = ((FC.STATE,), d[:, 0])
                else:
                    dvars[v] = (out_dims, d)

            ds = Dataset(coords=coords, data_vars=dvars)

            i0 = mdata.chunki_states
            t0 = mdata.chunki_points
            vrb = max(algo.verbosity - 1, 0)
            if out_dims == (FC.STATE, FC.TURBINE):
                fpath = out_dir / f"{base_name}_{i0:06d}.nc"
            else:
                fpath = out_dir / f"{base_name}_{i0:06d}_{t0:06d}.nc"
            write_nc_file(
                ds,
                fpath,
                nc_engine=config.nc_engine or "netcdf4",
                verbosity=vrb,
            )

        return results if ret_data else None

    def _write_ani(
        self,
        algo: Algorithm,
        chunk_key: tuple[int, int],
        write_chunk_ani: dict[str, Any] | None,
        *data: Any,
    ) -> None:
        """Helper function for optionally writing chunk flow animations to file"""
        if write_chunk_ani is not None:
            from foxes.output import write_chunk_ani_xy

            pars = write_chunk_ani.copy()
            chk = pars.pop("chunk")

            def _do_run(chk: Any) -> bool:
                if isinstance(chk, list):
                    for c in chk:
                        if _do_run(c):
                            return True
                    return False
                else:
                    return (
                        chk == chunk_key
                        if isinstance(chk, tuple)
                        else chk == chunk_key[0]
                    )

            if _do_run(chk):
                write_chunk_ani_xy(algo, *data, **pars)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the chunk calculation"""
        pass


class Engine(ABC):
    """
    Abstract base class for foxes calculation engines.

    Attributes
    ----------
    chunk_size_states
        The size of a state chunk.
    chunk_size_points
        The size of a point chunk.
    progress_bar
        Whether to use a progress bar instead of printing reached-percent
        updates. If ``None``, neither a progress bar nor progress messages are
        used.
    verbosity
        The verbosity level; ``0`` means silent.

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
        Construct the engine.

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
            Use a progress bar instead of printing reached-percent lines. If
            ``None``, neither the progress bar nor progress prints are used.
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
        self.__entered = True
        if has_engine():
            raise ValueError(
                f"Cannot enter engine '{self.name}', since engine already set to '{type(get_engine()).__name__}'"
            )
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
        if self.chunk_size_states is None and self.chunk_size_states is None:
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
        isel: dict[str, Any] | None = None,
        default_n_states: int | None = None,
    ) -> tuple[list[Any], int | None]:
        """
        Take subsets of datasets.

        Parameters
        ----------
        datasets
            The xarray dataset or data array objects.
        sel
            The selection dictionary.
        isel
            The index selection dictionary.
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
                    s = {c: u for c, u in sel.items() if c in data.coords}
                    new_datasets.append(data.sel(s) if len(s) else data)
                else:
                    new_datasets.append(data)
            subsets = new_datasets

        if isel is not None:
            new_datasets = []
            for data in subsets:
                if data is not None:
                    s = {c: u for c, u in isel.items() if c in data.dims}
                    new_datasets.append(data.isel(s) if len(s) > 0 else data)
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
        out_vars: list[str],
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
        out_vars
            Names of the output variables.
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

    def new_chunk_results_manager(
        self, algo: Algorithm, **kwargs: Any
    ) -> ChunkResultsManager:
        """
        Create a new chunk results manager.

        Parameters
        ----------
        algo
            The algorithm object.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        crm
            The chunk results manager.

        Example
        -------
        Derived engines should receive results from chunked calculations via

        >>> with engine.new_chunk_results_manager(...) as results_man:
        >>>     ...
        >>>     results_man.update(results, futures)
        >>>     ...

        After leaving the ``with`` block, the final results are available via
        ``results_man.results``.

        """
        return self.ChunkResultsManager(algo=algo, engine=self, **kwargs)

    class ChunkResultsManager:
        """Helper class for results management during chunk calculations"""

        def __init__(
            self,
            algo: Algorithm,
            engine: Engine,
            chunk_store: Any,
            goal_data: Dataset,
            n_chunks_states: int,
            n_chunks_targets: int,
            out_vars: list[str],
            out_dims: tuple[str, ...],
            coords: dict[str, Any],
            iterative: bool,
            write_nc: dict[str, Any] | None,
        ) -> None:
            """
            Construct the chunk results manager.

            Parameters
            ----------
            algo
                The algorithm object.
            engine
                The engine object.
            chunk_store
                The chunk store.
            goal_data
                The goal dataset.
            n_chunks_states
                The number of state chunks.
            n_chunks_targets
                The number of target chunks.
            out_vars
                The output variables.
            out_dims
                The output dimensions.
            coords
                The coordinates.
            iterative
                Whether the calculation is iterative.
            write_nc
                NetCDF output parameters, or ``None``.

            """
            self.algo = algo
            self.engine = engine
            self.chunk_store = chunk_store
            self.name = engine.name
            self.ci_states = 0
            self.ci_targets = 0
            self.counter = 0
            self.scount = 0
            self.wcount = 0
            self.wfutures: list[Any] = []
            self.fcounter = 0
            self.split_size = None
            self.pdone = -1
            self.pbar: Any = None
            self.res_vars: list[str] | None = None
            self.goal_data = goal_data
            self.data_vars: dict[str, Any] = {}
            self.out_dir: Any = None
            self.pack: bool | None = None
            self.base_name: str | None = None
            self.ret_data = True
            self.gen_size: Iterator[Any] | None = None
            self.write_on_fly = False
            self.write_from_ds = False
            self.n_chunks_states = n_chunks_states
            self.n_chunks_targets = n_chunks_targets
            self.n_chunks_all = n_chunks_states * n_chunks_targets
            self.out_dims = out_dims
            self.coords = coords
            self.out_vars = out_vars
            self.iterative = iterative
            self.tres: dict[str, list[np.ndarray]] | None = None
            self.verbosity = engine.verbosity
            self.results: Dataset | None = None

            # read parameters for file writing
            if write_nc is not None and not (iterative and not algo.final_iteration):
                self.out_dir = get_output_path(write_nc.get("out_dir", "."))
                self.base_name = write_nc["base_name"]
                self.ret_data = write_nc.get("ret_data", False)
                self.split_mode = write_nc.get("split", None)
                self.out_dir.mkdir(parents=True, exist_ok=True)
                self.pack = write_nc.get("pack", True)
                out_fpath = self.out_dir / (self.base_name + "_*.nc")
                if self.split_mode == "chunks":
                    self.engine.print(
                        f"{self.name}: Writing results to '{out_fpath}', using split = {self.split_mode}, ret_data = {self.ret_data}"
                    )
                elif self.split_mode == "input":
                    self.gen_size = algo.states.gen_states_split_size()
                    self.split_size = next(self.gen_size)
                elif isinstance(self.split_mode, int):
                    self.split_size = self.split_mode
                elif self.split_mode is None:
                    self.split_size = None
                else:
                    raise ValueError(
                        f"Invalid split mode '{self.split_mode}' in 'write_nc', expected 'chunks', 'input', int or None"
                    )
                if self.split_size is None:
                    out_fpath = self.out_dir / (self.base_name + ".nc")
                if self.split_mode != "chunks":
                    self.write_on_fly = (
                        not self.ret_data and self.split_size is not None
                    )
                    self.write_from_ds = not self.write_on_fly
                    self.ret_data = write_nc.get("ret_data", self.write_from_ds)
                    self.engine.print(
                        f"{self.name}: Writing results to '{out_fpath}', using split = {self.split_mode}, on_fly = {self.write_on_fly}, ret_data = {self.ret_data}"
                    )

            self.__entered = False

        def __enter__(self) -> Engine.ChunkResultsManager:
            if self.__entered:
                raise ValueError("Enter called for already entered ChunkResultsManager")
            self.__entered = True
            self.engine.print(
                self.engine.get_start_calc_message(
                    self.n_chunks_states, self.n_chunks_targets
                )
            )
            if self.verbosity > 0 and self.engine.has_progress_bar:
                self.pbar = tqdm(total=self.n_chunks_all)
            return self

        def _red_dims(
            self, data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
        ) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
            """Helper function for reducing dimensions of data vars"""
            dvars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
            for v, (dims, d) in data_vars.items():
                if (
                    dims == (FC.STATE, FC.TURBINE)
                    and d.shape[1] == 1
                    and self.algo.n_turbines > 1
                ):
                    dvars[v] = ((FC.STATE,), d[:, 0])
                elif (
                    dims == (FC.STATE, FC.TARGET, FC.TPOINT)
                    and self.goal_data.sizes[FC.TARGET] > self.n_chunks_targets
                    and d.shape[1:] == (self.n_chunks_targets, 1)
                ):
                    dvars[v] = ((FC.STATE,), d[:, 0, 0])
                else:
                    dvars[v] = (dims, d)
            return dvars

        def _write_parts_on_fly(self, futures: list[Any] | None) -> None:
            """Helper function for writing results to files on the fly"""
            vrb = max(self.verbosity - 1, 0)
            wfutures: list[Any] = []
            n_states = self.algo.n_states
            assert n_states is not None
            if self.split_size is not None and self.split_size > 0:
                assert self.out_dir is not None
                assert self.base_name is not None
                splits = min(self.split_size, n_states - self.wcount)
                while (
                    n_states - self.wcount > 0 and self.scount - self.wcount >= splits
                ):
                    for v in self.data_vars.keys():
                        if len(self.data_vars[v][1]) > 1:
                            self.data_vars[v][1] = [
                                np.concatenate(self.data_vars[v][1], axis=0)
                            ]

                    dvars = {
                        v: (d[0], d[1][0][:splits]) for v, d in self.data_vars.items()
                    }
                    dvars = self._red_dims(dvars)
                    crds = {v: d for v, d in self.coords.items()}
                    crds[FC.STATE] = self.coords[FC.STATE][
                        self.wcount : self.wcount + splits
                    ]
                    ds = Dataset(coords=crds, data_vars=dvars)
                    del dvars, crds

                    if self.scount - self.wcount == splits:
                        for v in self.data_vars.keys():
                            self.data_vars[v][1] = []
                    else:
                        for v in self.data_vars.keys():
                            self.data_vars[v][1] = [self.data_vars[v][1][0][splits:]]

                    fpath = self.out_dir / f"{self.base_name}_{self.fcounter:06d}.nc"
                    if futures is not None and len(futures) < self.engine.n_workers:
                        future = self.engine.submit(
                            write_nc_file,
                            ds,
                            fpath,
                            nc_engine=config.nc_engine or "netcdf4",
                            verbosity=vrb,
                            pack=self.pack if self.pack is not None else False,
                        )
                        wfutures.append(future)
                        del future
                    else:
                        write_nc_file(
                            ds,
                            fpath,
                            nc_engine=config.nc_engine or "netcdf4",
                            verbosity=vrb,
                            pack=self.pack if self.pack is not None else False,
                        )
                    del ds

                    self.wcount += splits
                    self.fcounter += 1

                    if n_states - self.wcount > 0:
                        if self.split_mode == "input":
                            try:
                                assert self.gen_size is not None
                                self.split_size = next(self.gen_size)
                            except StopIteration:
                                self.split_size = n_states - self.wcount
                        splits = min(self.split_size, n_states - self.wcount)

            self.wfutures += wfutures

        def update(
            self, results: dict[tuple[int, int], Any], futures: list[Any] | None = None
        ) -> None:
            """
            Update chunk calculation progress and accumulate the results.

            Parameters
            ----------
            results
                A dictionary of chunk results.
            futures
                The current futures for asynchronous writing, or ``None``.

            """
            assert self.__entered, (
                "ChunkResultsManager: update_chunk_progress called without enter"
            )

            chunk_key = (self.ci_states, self.ci_targets)
            while chunk_key in results:
                r, cstore = results.pop(chunk_key)

                for k, c in cstore.items():
                    if k in self.chunk_store:
                        self.chunk_store[k].update(c)
                    else:
                        self.chunk_store[k] = c

                if r is not None:
                    if self.res_vars is None:
                        self.res_vars = list(r.keys())
                        for v in self.out_vars:
                            if v in self.res_vars:
                                self.data_vars[v] = [self.out_dims, []]
                            else:
                                self.data_vars[v] = (
                                    self.goal_data[v].dims,
                                    self.goal_data[v].to_numpy(),
                                )

                    if self.n_chunks_targets == 1:
                        for v in self.res_vars:
                            if v in self.data_vars:
                                self.data_vars[v][1].append(r[v])
                        self.scount += r[self.res_vars[0]].shape[0]

                    else:
                        if self.tres is None:
                            self.tres = {v: [] for v in self.res_vars}
                        for v in self.res_vars:
                            self.tres[v].append(r[v])
                        if self.ci_targets == self.n_chunks_targets - 1:
                            found = False
                            for v in self.res_vars:
                                if v in self.data_vars:
                                    self.data_vars[v][1].append(
                                        np.concatenate(self.tres[v], axis=1)
                                    )
                                    if not found and self.write_on_fly:
                                        self.scount += self.data_vars[v][1][-1].shape[0]
                                    found = True
                            self.tres = None

                    if self.write_on_fly:
                        self._write_parts_on_fly(futures)

                self.counter += 1
                if self.pbar is not None:
                    self.pbar.update()
                elif self.verbosity > 0 and self.engine.prints_progress:
                    pr = int(100 * self.counter / self.n_chunks_all)
                    if pr > self.pdone:
                        self.pdone = pr
                        print(
                            f"{self.name}: Completed {self.counter} of {self.n_chunks_all} chunks, {self.pdone}%"
                        )

                self.ci_targets += 1
                if self.ci_targets >= self.n_chunks_targets:
                    self.ci_targets = 0
                    self.ci_states += 1
                chunk_key = (self.ci_states, self.ci_targets)

        def __exit__(self, *exit_args: Any) -> None:
            assert self.__entered, "ChunkResultsManager: exit called without enter"
            assert self.counter == self.n_chunks_all, (
                f"{self.name}: Incomplete chunk calculation: {self.counter} of {self.n_chunks_all} chunks done"
            )
            assert self.ci_states == self.n_chunks_states, (
                f"{self.name}: Incomplete chunk calculation: only {self.ci_states} of {self.n_chunks_states} states chunks done"
            )

            if self.wfutures is not None:
                for wf in self.wfutures:
                    self.engine.await_result(wf)

            if self.pbar is not None:
                self.pbar.close()
            self.engine.print(
                f"{self.name}: Completed all {self.n_chunks_all} chunks\n"
            )

            vrb = max(self.verbosity - 1, 0)
            if self.ret_data or self.write_from_ds:
                assert self.res_vars is not None
                for v in self.res_vars:
                    if v in self.data_vars:
                        if len(self.data_vars[v][1]) > 1:
                            self.data_vars[v][1] = np.concatenate(
                                self.data_vars[v][1], axis=0
                            )
                        elif len(self.data_vars[v][1]) == 1:
                            self.data_vars[v][1] = self.data_vars[v][1][0]
                self.data_vars = self._red_dims(self.data_vars)
                self.results = Dataset(
                    coords=self.coords,
                    data_vars=self.data_vars,
                )

                if self.write_from_ds:
                    assert self.out_dir is not None
                    assert self.base_name is not None
                    if self.split_size is None:
                        fpath = self.out_dir / f"{self.base_name}.nc"
                        write_nc_file(
                            self.results,
                            fpath,
                            nc_engine=config.nc_engine or "netcdf4",
                            verbosity=vrb,
                        )
                    else:
                        wcount = 0
                        fcounter = 0
                        wfutures: list[Any] = []
                        n_states = self.algo.n_states
                        assert n_states is not None
                        while wcount < n_states:
                            splits = min(self.split_size, n_states - wcount)
                            assert self.results is not None
                            dssub = self.results.isel(
                                {FC.STATE: slice(wcount, wcount + splits)}
                            )

                            fpath = self.out_dir / f"{self.base_name}_{fcounter:06d}.nc"
                            future = self.engine.submit(
                                write_nc_file,
                                dssub,
                                fpath,
                                nc_engine=config.nc_engine or "netcdf4",
                                verbosity=vrb,
                            )
                            wfutures.append(future)
                            del dssub, future

                            wcount += splits
                            fcounter += 1

                            if wcount < n_states and self.split_mode == "input":
                                try:
                                    assert self.gen_size is not None
                                    self.split_size = next(self.gen_size)
                                except StopIteration:
                                    self.split_size = n_states - wcount
                        for wf in wfutures:
                            self.engine.await_result(wf)

            del (
                self.ci_states,
                self.ci_targets,
                self.counter,
                self.scount,
                self.wcount,
                self.wfutures,
                self.fcounter,
                self.split_size,
                self.pdone,
                self.pbar,
                self.res_vars,
                self.data_vars,
                self.goal_data,
                self.out_dir,
                self.base_name,
                self.ret_data,
                self.gen_size,
                self.write_on_fly,
                self.write_from_ds,
                self.out_dims,
                self.coords,
                self.out_vars,
                self.iterative,
                self.tres,
            )
            self.__entered = False

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


def launch_parallel_calc(self: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Launch a parallel calculation using the active engine.

    Parameters
    ----------
    args
        Additional parameters for running the calculation.
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
        results = engine.run_calculation(self, *args, **kwargs)
    else:
        with Engine.new("default") as e:
            results = e.run_calculation(self, *args, **kwargs)
    return results
