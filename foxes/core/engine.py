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

    def _get_start_calc_message(
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

    def start_chunk_calculation(
        self,
        algo,
        coords,
        goal_data,
        n_chunks_states,
        n_chunks_targets,
        iterative,
        write_nc,
    ):
        """
        Marks the start of a chunk calculation

        Parameters
        ----------
        n_chunks_states: int
            The number of states chunks
        n_chunks_targets: int
            The number of targets chunks

        """
        assert not self.__running_chunk_calc, f"{self.name}: Chunk calculation already running"

        # prepare:
        self._ci_states = 0
        self._ci_targets = 0
        self._counter = 0
        self._scount = 0
        self._wcount = 0
        self._wfutures = []
        self._fcounter = 0
        self._split_size = None
        self._pdone = -1
        self._pbar = None
        self._res_vars = None
        self._goal_data = goal_data
        self._data_vars = {}
        self._out_dir = None
        self._base_name = None
        self._ret_data = True
        self._gen_size = None
        self._write_on_fly = False
        self._write_from_ds = False
        self._n_chunks_states = n_chunks_states
        self._n_chunks_targets = n_chunks_targets
        self._n_chunks_all = n_chunks_states * n_chunks_targets
        self._coords = coords
        self._iterative = iterative
        self._tres = None

        # read parameters for file writing
        if write_nc is not None and not (iterative and not algo.final_iteration):
            self._out_dir = get_output_path(write_nc.get("out_dir", "."))
            self._base_name = write_nc["base_name"]
            self._ret_data = write_nc.get("ret_data", False)
            self._split_mode = write_nc.get("split", None)
            self._out_dir.mkdir(parents=True, exist_ok=True)
            out_fpath = self._out_dir / (self._base_name + "_*.nc")
            if self._split_mode == "chunks":
                self.print(
                    f"{self.name}: Writing results to '{out_fpath}', using split = {self._split_mode}, ret_data = {self._ret_data}"
                )
            elif self._split_mode == "input":
                self._gen_size = algo.states.gen_states_split_size()
                self._split_size = next(self._gen_size)
            elif isinstance(self._split_mode, int):
                self._split_size = self._split_mode
            elif self._split_mode is None:
                self._split_size = None
            else:
                raise ValueError(
                    f"Invalid split mode '{self._split_mode}' in 'write_nc', expected 'chunks', 'input', int or None"
                )
            if self._split_size is None:
                out_fpath = self._out_dir / (self._base_name + ".nc")
            if self._split_mode != "chunks":
                self._write_on_fly = not self._ret_data and self._split_size is not None
                self._write_from_ds = not self._write_on_fly
                self._ret_data = write_nc.get("ret_data", self._write_from_ds)
                self.print(
                    f"{self.name}: Writing results to '{out_fpath}', using split = {self._split_mode}, on_fly = {self._write_on_fly}, ret_data = {self._ret_data}"
                )

        self.print(self._get_start_calc_message(n_chunks_states, n_chunks_targets))

        if self.verbosity > 0 and self.has_progress_bar:
            self._pbar = tqdm(total=self._n_chunks_all)

        self.__running_chunk_calc = True

    def _red_dims(self, algo, data_vars):
        """Helper function for reducing unneccessary dimensions of data vars"""
        dvars = {}
        for v, (dims, d) in data_vars.items():
            if (
                dims == (FC.STATE, FC.TURBINE)
                and d.shape[1] == 1
                and algo.n_turbines > 1
            ):
                dvars[v] = ((FC.STATE,), d[:, 0])
            elif (
                dims == (FC.STATE, FC.TARGET, FC.TPOINT)
                and self._goal_data.sizes[FC.TARGET] > self._n_chunks_targets
                and d.shape[1:] == (self._n_chunks_targets, 1)
            ):
                dvars[v] = ((FC.STATE,), d[:, 0, 0])
            else:
                dvars[v] = (dims, d)
        return dvars

    def _write_parts_on_fly(self, algo, futures):
        """Helper function for writing chunked results to netCDF,
        without ever constructing the full Dataset in memory
        """
        vrb = max(self.verbosity - 1, 0)
        wfutures = []
        if self._split_size is not None:
            splits = min(self._split_size, algo.n_states - self._wcount)
            while algo.n_states - self._wcount > 0 and self._scount - self._wcount >= splits:
                for v in self._data_vars.keys():
                    if len(self._data_vars[v][1]) > 1:
                        self._data_vars[v][1] = [np.concatenate(self._data_vars[v][1], axis=0)]

                dvars = {v: (d[0], d[1][0][:splits]) for v, d in self._data_vars.items()}
                dvars = self._red_dims(algo, dvars)
                crds = {v: d for v, d in self._coords.items()}
                crds[FC.STATE] = self._coords[FC.STATE][self._wcount : self._wcount + splits]
                ds = Dataset(coords=crds, data_vars=dvars)
                del dvars, crds

                if self._scount - self._wcount == splits:
                    for v in self._data_vars.keys():
                        self._data_vars[v][1] = []
                else:
                    for v in self._data_vars.keys():
                        self._data_vars[v][1] = [self._data_vars[v][1][0][splits:]]

                fpath = self._out_dir / f"{self._base_name}_{self._fcounter:06d}.nc"
                args = (ds, fpath)
                kwargs = dict(nc_engine=config.nc_engine, verbosity=vrb)
                if futures is not None and len(futures) < self.n_workers:
                    future = self.submit(write_nc_file, *args, **kwargs)
                    wfutures.append(future)
                    del future
                else:
                    write_nc_file(*args, **kwargs)
                del ds, args, kwargs

                self._wcount += splits
                self._fcounter += 1

                if algo.n_states - self._wcount > 0:
                    if self._split_mode == "input":
                        try:
                            self._split_size = next(self._gen_size)
                        except StopIteration:
                            self._split_size = algo.n_states - self._wcount
                    splits = min(self._split_size, algo.n_states - self._wcount)

        self._wfutures += wfutures

    def update_chunk_progress(
        self,
        algo,
        results,
        out_coords,
        goal_data,
        out_vars,
        futures=None,
    ):
        """
        Updates the chunk calculation progress
        """
        assert self.__running_chunk_calc, f"{self.name}: No chunk calculation running"

        chunk_key = (self._ci_states, self._ci_targets)
        while chunk_key in results:
            r, cstore = results.pop(chunk_key)

            if self._iterative:
                for k, c in cstore.items():
                    if k in algo.chunk_store:
                        algo.chunk_store[k].update(c)
                    else:
                        algo.chunk_store[k] = c

            if r is not None:

                if self._res_vars is None:
                    self._res_vars = list(r.keys())
                    for v in out_vars:
                        if v in self._res_vars:
                            self._data_vars[v] = [out_coords, []]
                        else:
                            self._data_vars[v] = (goal_data[v].dims, goal_data[v].to_numpy())

                if self._n_chunks_targets == 1:
                    for v in self._res_vars:
                        if v in self._data_vars:
                            self._data_vars[v][1].append(r[v])
                    self._scount += r[self._res_vars[0]].shape[0]

                else:
                    if self._tres is None:
                        self._tres = {v: [] for v in self._res_vars}
                    for v in self._res_vars:
                        self._tres[v].append(r[v])
                    if self._ci_targets == self._n_chunks_targets - 1:
                        found = False
                        for v in self._res_vars:
                            if v in self._data_vars:
                                self._data_vars[v][1].append(np.concatenate(self._tres[v], axis=1))
                                if not found and self._write_on_fly:
                                    self._scount += self._data_vars[v][1][-1].shape[0]
                                found = True
                        self._tres = None

                if self._write_on_fly:
                    self._write_parts_on_fly(algo, futures)

            self._counter += 1
            if self._pbar is not None:
                self._pbar.update()
            elif self.verbosity > 0 and self.prints_progress:
                pr = int(100 * self._counter / self._n_chunks_all)
                if pr > self._pdone:
                    self._pdone = pr
                    print(f"{self.name}: Completed {self._counter} of {self._n_chunks_all} chunks, {self._pdone}%")

            self._ci_targets += 1
            if self._ci_targets >= self._n_chunks_targets:
                self._ci_targets = 0
                self._ci_states += 1
            chunk_key = (self._ci_states, self._ci_targets)

    def end_chunk_calculation(self, algo):
        """
        Marks the end of a chunk calculation
        """
        assert self.__running_chunk_calc, f"{self.name}: No chunk calculation running"
        assert self._counter == self._n_chunks_all, (
            f"{self.name}: Incomplete chunk calculation: {self._counter} of {self._n_chunks_all} chunks done"
        )
        assert self._ci_states == self._n_chunks_states, (
            f"{self.name}: Incomplete chunk calculation: only {self._ci_states} of {self._n_chunks_states} states chunks done"  
        )

        if self._wfutures is not None:
            for wf in self._wfutures:
                self.await_result(wf)

        if self._pbar is not None:
            self._pbar.close()
        self.print(f"{self.name}: Completed all {self._n_chunks_all} chunks\n")

        vrb = max(self.verbosity - 1, 0)
        ds = None
        if self._ret_data or self._write_from_ds:
            for v in self._res_vars:
                if v in self._data_vars:
                    if len(self._data_vars[v][1]) > 1:
                        self._data_vars[v][1] = np.concatenate(self._data_vars[v][1], axis=0)
                    elif len(self._data_vars[v][1]) == 1:
                        self._data_vars[v][1] = self._data_vars[v][1][0]
            self._data_vars = self._red_dims(algo, self._data_vars)
            ds = Dataset(
                coords=self._coords,
                data_vars=self._data_vars,
            )

            if self._write_from_ds:
                if self._split_size is None:
                    fpath = self._out_dir / f"{self._base_name}.nc"
                    write_nc_file(
                        ds, fpath, nc_engine=config.nc_engine, verbosity=vrb
                    )
                else:
                    wcount = 0
                    fcounter = 0
                    wfutures = []
                    while wcount < algo.n_states:
                        splits = min(self._split_size, algo.n_states - wcount)
                        dssub = ds.isel({FC.STATE: slice(wcount, wcount + splits)})

                        fpath = self._out_dir / f"{self._base_name}_{fcounter:06d}.nc"
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

                        if wcount < algo.n_states and self._split_mode == "input":
                            try:
                                self._split_size = next(self._gen_size)
                            except StopIteration:
                                self._split_size = algo.n_states - wcount

                    for wf in wfutures:
                        self.await_result(wf)

        self._ci_states = None
        self._ci_targets = None
        self._counter = None
        self._scount = None
        self._wcount = None
        self._wfutures = None
        self._fcounter = None
        self._split_size = None
        self._pdone = None
        self._pbar = None
        self._res_vars = None
        self._data_vars = None
        self._goal_data = None
        self._out_dir = None
        self._base_name = None
        self._ret_data = None
        self._gen_size = None
        self._write_on_fly = None
        self._write_from_ds = None
        self._coords = None
        self._iterative = None
        self._tres = None
        self.__running_chunk_calc = False

        return ds

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
