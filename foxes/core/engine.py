import os
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from xarray import Dataset

from foxes.config import config, get_output_path
from foxes.utils import new_instance
from foxes.utils import write_nc as write_nc_file
import foxes.constants as FC

from .data import MData, FData, TData

__global_engine_data__ = dict(
    engine=None,
)


class Engine(ABC):
    """
    Abstract base clas for foxes calculation engines

    Attributes
    ----------
    chunk_size_states: int
        The size of a states chunk
    chunk_size_points: int
        The size of a points chunk
    progress_bar: bool, optional
        Use a progress bar instead of simply
        printing lines of reached percentages.
        Unless progress_bar is None, then neither
    verbosity: int
        The verbosity level, 0 = silent

    :group: core

    """

    def __init__(
        self,
        chunk_size_states=None,
        chunk_size_points=None,
        n_procs=None,
        progress_bar=True,
        verbosity=1,
    ):
        """
        Constructor.

        Parameters
        ----------
        chunk_size_states: int, optional
            The size of a states chunk
        chunk_size_points: int, optional
            The size of a points chunk
        n_procs: int, optional
            The number of processes to be used,
            or None for automatic
        progress_bar: bool, optional
            Use a progress bar instead of simply
            printing lines of reached percentages.
            Unless progress_bar is None, then neither
        verbosity: int
            The verbosity level, 0 = silent

        """
        self.chunk_size_states = chunk_size_states
        self.chunk_size_points = chunk_size_points
        self.progress_bar = progress_bar
        self.verbosity = verbosity

        try:
            self._n_procs = n_procs if n_procs is not None else os.process_cpu_count()
        except AttributeError:
            self._n_procs = os.cpu_count()
        self._n_workers = max(self._n_procs - 1, 1)

        self.__name = type(self).__name__
        self.__initialized = False
        self.__entered = False
        self.__running_chunk_calc = False

    def __repr__(self):
        s = f"n_procs={self.n_procs}, chunk_size_states={self.chunk_size_states}, chunk_size_points={self.chunk_size_points}"
        return f"{self.name}({s})"

    def __enter__(self):
        if self.__entered:
            raise ValueError("Enter called for already entered engine")
        self.__entered = True
        if not self.initialized:
            self.initialize()
        return self

    def __exit__(self, *exit_args):
        if not self.__entered:
            raise ValueError("Exit called for not entered engine")
        self.__entered = False
        if self.initialized:
            self.finalize(*exit_args)

    def __del__(self):
        if self.initialized:
            self.finalize()

    @property
    def name(self):
        """
        The engine's name

        Returns
        -------
        nme: str
            The engine's name

        """
        return self.__name

    @property
    def n_procs(self):
        """
        The number of processes

        Returns
        -------
        n_procs: int
            The number of processes

        """
        return self._n_procs

    @property
    def n_workers(self):
        """
        The number of worker processes

        Returns
        -------
        n_workers: int
            The number of worker processes

        """
        return self._n_workers

    @property
    def has_progress_bar(self):
        """
        Flag for active progress bar

        Returns
        -------
        has_pbar: bool
            True if progress bar is active

        """
        return self.progress_bar is not None and self.progress_bar

    @property
    def prints_progress(self):
        """
        Flag for active progress printing

        Returns
        -------
        has_pbar: bool
            True if progress printing is active

        """
        return self.progress_bar is not None and not self.progress_bar

    @property
    def entered(self):
        """
        Flag that this model has been entered.

        Returns
        -------
        flag: bool
            True if the model has been entered.

        """
        return self.__entered

    @property
    def initialized(self):
        """
        Initialization flag.

        Returns
        -------
        flag: bool
            True if the model has been initialized.

        """
        return self.__initialized

    @property
    def running_chunk_calc(self):
        """
        Flag that a chunk calculation is running.

        Returns
        -------
        flag: bool
            True if a chunk calculation is running.

        """
        return self.__running_chunk_calc

    def initialize(self):
        """
        Initializes the engine.
        """
        if not self.entered:
            self.__enter__()
        elif not self.initialized:
            if get_engine(error=False, default=False) is not None:
                raise ValueError(
                    f"Cannot initialize engine '{self.name}', since engine already set to '{type(get_engine()).__name__}'"
                )
            __global_engine_data__["engine"] = self
            self.__initialized = True

    def finalize(self, type=None, value=None, traceback=None):
        """
        Finalizes the engine.

        Parameters
        ----------
        type: object, optional
            Dummy argument for the exit function
        value: object, optional
            Dummy argument for the exit function
        traceback: object, optional
            Dummy argument for the exit function

        """
        if self.entered:
            self.__exit__(type, value, traceback)
        elif self.initialized:
            __global_engine_data__["engine"] = None
            self.__initialized = False

    def print(self, *args, level=1, **kwargs):
        """Prints based on verbosity"""
        if self.verbosity >= level:
            print(*args, **kwargs)

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def map(
        self,
        func,
        inputs,
        *args,
        **kwargs,
    ):
        """
        Runs a function on a list of files

        Parameters
        ----------
        func: Callable
            Function to be called on each file,
            func(input, *args, **kwargs) -> data
        inputs: array-like
            The input data list
        args: tuple, optional
            Arguments for func
        kwargs: dict, optional
            Keyword arguments for func

        Returns
        -------
        results: list
            The list of results

        """
        pass

    @property
    def loop_dims(self):
        """
        Gets the loop dimensions (possibly chunked)

        Returns
        -------
        dims: list of str
            The loop dimensions (possibly chunked)

        """
        if self.chunk_size_states is None and self.chunk_size_states is None:
            return []
        elif self.chunk_size_states is None:
            return [FC.TARGET]
        elif self.chunk_size_points is None:
            return [FC.STATE]
        else:
            return [FC.STATE, FC.TARGET]

    def select_subsets(self, *datasets, sel=None, isel=None):
        """
        Takes subsets of datasets

        Parameters
        ----------
        datasets: tuple
            The xarray.Dataset or xarray.Dataarray objects
        sel: dict, optional
            The selection dictionary
        isel: dict, optional
            The index selection dictionary

        Returns
        -------
        subsets: list
            The subsets of the input data

        """
        if sel is not None:
            new_datasets = []
            for data in datasets:
                if data is not None:
                    s = {c: u for c, u in sel.items() if c in data.coords}
                    new_datasets.append(data.sel(s) if len(s) else data)
                else:
                    new_datasets.append(data)
            datasets = new_datasets

        if isel is not None:
            new_datasets = []
            for data in datasets:
                if data is not None:
                    s = {c: u for c, u in isel.items() if c in data.coords}
                    new_datasets.append(data.isel(s) if len(s) else data)
                else:
                    new_datasets.append(data)
            datasets = new_datasets

        return datasets

    def calc_chunk_sizes(self, n_states, n_targets=1):
        """
        Computes the sizes of states and points chunks

        Parameters
        ----------
        n_states: int
            The number of states
        n_targets: int
            The number of point targets

        Returns
        -------
        chunk_sizes_states: numpy.ndarray
            The sizes of all states chunks, shape: (n_chunks_states,)
        chunk_sizes_targets: numpy.ndarray
            The sizes of all targets chunks, shape: (n_chunks_targets,)

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
        chunk_sizes_targets = [n_targets]
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
            chunk_sizes_targets = np.full(n_chunks_targets, chunk_size_targets)
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
        algo,
        model_data,
        farm_data,
        point_data,
        states_i0_i1,
        targets_i0_i1,
        out_vars,
        chunki_states,
        chunki_points,
        n_chunks_states,
        n_chunks_points,
    ):
        """
        Extracts the data for a single chunk calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset
            The initial farm data
        point_data: xarray.Dataset
            The initial point data
        states_i0_i1: tuple
            The (start, end) values of the states
        targets_i0_i1: tuple
            The (start, end) values of the targets
        out_vars: list of str
            Names of the output variables
        chunki_states: int
            The index of the states chunk
        chunki_points: int
            The index of the points chunk
        n_chunks_states: int
            The number of states chunks
        n_chunks_points: int
            The number of points chunks

        Returns
        -------
        data: tuple of foxes.core.Data
            The input data for the chunk calculation,
            either (mdata, fdata) or (mdata, fdata, tdata)

        """
        # prepare:
        i0_states, i1_states = states_i0_i1
        i0_targets, i1_targets = targets_i0_i1
        s_states = np.s_[i0_states:i1_states]
        s_targets = np.s_[i0_targets:i1_targets]

        # create mdata:
        mdata = MData.from_dataset(
            model_data,
            s_states=s_states,
            loop_dims=[FC.STATE],
            states_i0=i0_states,
            copy=True,
            chunki_states=chunki_states,
            chunki_points=chunki_points,
            n_chunks_states=n_chunks_states,
            n_chunks_points=n_chunks_points,
        )

        # create fdata:
        if farm_data is not None:
            fdata = FData.from_dataset(
                farm_data,
                mdata=mdata,
                s_states=s_states,
                callback=None,
                states_i0=i0_states,
                copy=True,
            )
        else:
            fdata = FData.from_mdata(
                mdata=mdata,
                states_i0=i0_states,
            )

        # create tdata:
        tdata = (
            TData.from_dataset(
                point_data,
                mdata=mdata,
                s_states=s_states,
                s_targets=s_targets,
                callback=None,
                states_i0=i0_states,
                copy=True,
            )
            if point_data is not None
            else None
        )

        return (mdata, fdata) if tdata is None else (mdata, fdata, tdata)

    def get_start_calc_message(
        self,
        n_chunks_states,
        n_chunks_targets,
    ):
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
        algo,
        model,
        model_data=None,
        farm_data=None,
        point_data=None,
    ):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel, optional
            The model that whose calculate function
            should be run
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset, optional
            The initial farm data
        point_data: xarray.Dataset, optional
            The initial point data

        Returns
        -------
        results: xarray.Dataset
            The model results

        """
        n_states = model_data.sizes[FC.STATE]
        if point_data is None:
            self.print(
                f"{self.name}: Calculating {n_states} states for {algo.n_turbines} turbines"
            )
        else:
            self.print(
                f"{self.name}: Calculating data at {point_data.sizes[FC.TARGET]} points for {n_states} states"
            )
        if not self.initialized:
            raise ValueError(f"Engine '{self.name}' not initialized")
        if not model.initialized:
            raise ValueError(f"Model '{model.name}' not initialized")

    def new_chunk_results_manager(self, algo, **kwargs):
        """
        Creates a new ChunkResultsManager

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        kwargs: dict, optional
            Additional keyword arguments

        Returns
        -------
        crm: foxes.core.engine.ChunkResultsManager
            The chunk results manager

        Example
        -------
        Derived engines should receive results from chunked calculations
        through
        ```python
        with engine.new_chunk_results_manager(...) as results_man:
            ...
            results_man.update(results, futures)
            ...
        ```
        After exiting the with-block, the final results are available
        through `results_man.results`.

        """
        return self.ChunkResultsManager(algo=algo, engine=self, **kwargs)

    class ChunkResultsManager:
        """Helper class for results management during chunk calculations"""

        def __init__(
            self,
            algo,
            engine,
            goal_data,
            n_chunks_states,
            n_chunks_targets,
            out_vars,
            out_dims,
            coords,
            iterative,
            write_nc,
        ):
            """
            Constructor

            Parameters
            ----------
            algo: foxes.core.Algorithm
                The algorithm object
            engine: foxes.core.Engine
                The engine object
            goal_data: xarray.Dataset
                The goal data
            n_chunks_states: int
                Number of state chunks
            n_chunks_targets: int
                Number of target chunks
            out_vars: list
                List of output variables
            out_dims: list
                List of output dimensions
            coords: dict
                Coordinates
            iterative: bool
                Whether the calculation is iterative
            write_nc: dict or None
                Write netCDF parameters

            """
            self.algo = algo
            self.engine = engine
            self.name = engine.name
            self.ci_states = 0
            self.ci_targets = 0
            self.counter = 0
            self.scount = 0
            self.wcount = 0
            self.wfutures = []
            self.fcounter = 0
            self.split_size = None
            self.pdone = -1
            self.pbar = None
            self.res_vars = None
            self.goal_data = goal_data
            self.data_vars = {}
            self.out_dir = None
            self.base_name = None
            self.ret_data = True
            self.gen_size = None
            self.write_on_fly = False
            self.write_from_ds = False
            self.n_chunks_states = n_chunks_states
            self.n_chunks_targets = n_chunks_targets
            self.n_chunks_all = n_chunks_states * n_chunks_targets
            self.out_dims = out_dims
            self.coords = coords
            self.out_vars = out_vars
            self.iterative = iterative
            self.tres = None
            self.verbosity = engine.verbosity
            self.results = None

            # read parameters for file writing
            if write_nc is not None and not (iterative and not algo.final_iteration):
                self.out_dir = get_output_path(write_nc.get("out_dir", "."))
                self.base_name = write_nc["base_name"]
                self.ret_data = write_nc.get("ret_data", False)
                self.split_mode = write_nc.get("split", None)
                self.out_dir.mkdir(parents=True, exist_ok=True)
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

        def __enter__(self):
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

        def _red_dims(self, data_vars):
            """Helper function for reducing dimensions of data vars"""
            dvars = {}
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

        def _write_parts_on_fly(self, futures):
            """Helper function for writing results to files on the fly"""
            vrb = max(self.verbosity - 1, 0)
            wfutures = []
            if self.split_size is not None:
                splits = min(self.split_size, self.algo.n_states - self.wcount)
                while (
                    self.algo.n_states - self.wcount > 0
                    and self.scount - self.wcount >= splits
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
                    args = (ds, fpath)
                    kwargs = dict(nc_engine=config.nc_engine, verbosity=vrb)
                    if futures is not None and len(futures) < self.engine.n_workers:
                        future = self.engine.submit(write_nc_file, *args, **kwargs)
                        wfutures.append(future)
                        del future
                    else:
                        write_nc_file(*args, **kwargs)
                    del ds, args, kwargs

                    self.wcount += splits
                    self.fcounter += 1

                    if self.algo.n_states - self.wcount > 0:
                        if self.split_mode == "input":
                            try:
                                self.split_size = next(self.gen_size)
                            except StopIteration:
                                self.split_size = self.algo.n_states - self.wcount
                        splits = min(self.split_size, self.algo.n_states - self.wcount)

            self.wfutures += wfutures

        def update(self, results, futures=None):
            """
            Updates the chunk calculation progress, adds results to data_vars

            Parameters
            ----------
            results: dict
                Dictionary of chunk results
            futures: list or None
                List of current futures for asynchronous writing

            """
            assert self.__entered, (
                "ChunkResultsManager: update_chunk_progress called without enter"
            )

            chunk_key = (self.ci_states, self.ci_targets)
            while chunk_key in results:
                r, cstore = results.pop(chunk_key)

                if self.iterative:
                    for k, c in cstore.items():
                        if k in self.algo.chunk_store:
                            self.algo.chunk_store[k].update(c)
                        else:
                            self.algo.chunk_store[k] = c

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

        def __exit__(self, *exit_args):
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
                    if self.split_size is None:
                        fpath = self.out_dir / f"{self.base_name}.nc"
                        write_nc_file(
                            self.results,
                            fpath,
                            nc_engine=config.nc_engine,
                            verbosity=vrb,
                        )
                    else:
                        wcount = 0
                        fcounter = 0
                        wfutures = []
                        while wcount < self.algo.n_states:
                            splits = min(self.split_size, self.algo.n_states - wcount)
                            dssub = self.results.isel(
                                {FC.STATE: slice(wcount, wcount + splits)}
                            )

                            fpath = self.out_dir / f"{self.base_name}_{fcounter:06d}.nc"
                            future = self.submit(
                                write_nc_file,
                                dssub,
                                fpath,
                                nc_engine=config.nc_engine,
                                verbosity=vrb,
                            )
                            wfutures.append(future)
                            del dssub, future

                            wcount += splits
                            fcounter += 1

                            if (
                                wcount < self.algo.n_states
                                and self.split_mode == "input"
                            ):
                                try:
                                    self.split_size = next(self.gen_size)
                                except StopIteration:
                                    self.split_size = self.algo.n_states - wcount
                        for wf in wfutures:
                            self.await_result(wf)

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
    def new(cls, engine_type, *args, **kwargs):
        """
        Run-time engine factory.

        Parameters
        ----------
        engine_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

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

        return new_instance(cls, engine_type, *args, **kwargs)


def get_engine(error=True, default=True):
    """
    Gets the global calculation engine

    Parameters
    ----------
    error: bool
        Flag for raising ValueError if no
        engine is found
    default: bool or dict or Engine
        Set default engine if not set yet

    Returns
    -------
    engine: foxes.core.Engine
        The foxes calculation engine

    :group: core

    """
    engine = __global_engine_data__["engine"]
    if engine is None:
        if isinstance(default, dict):
            engine = Engine.new(**default)
            print(f"Selecting default engine '{engine}'")
            engine.initialize()
            return engine
        elif isinstance(default, Engine):
            print(f"Selecting default engine '{default}'")
            default.initialize()
            return default
        elif isinstance(default, bool) and default:
            engine = Engine.new(engine_type="DefaultEngine", verbosity=1)
            print(f"Selecting '{engine}'")
            engine.initialize()
            return engine
        elif error:
            raise ValueError("Engine not found.")
    return engine


def has_engine():
    """
    Flag that checks if engine has been set

    Returns
    -------
    flag: bool
        True if engine has been set

    :group: core

    """
    return __global_engine_data__["engine"] is not None


def reset_engine():
    """
    Resets the global calculation engine

    :group: core

    """
    engine = get_engine(error=False, default=False)
    if engine is not None:
        engine.finalize(type=None, value=None, traceback=None)
