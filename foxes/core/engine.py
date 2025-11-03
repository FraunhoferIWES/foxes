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
    n_procs: int, optional
        The number of processes to be used,
        or None for automatic
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
        try:
            self.n_procs = n_procs if n_procs is not None else os.process_cpu_count()
        except AttributeError:
            self.n_procs = os.cpu_count()
        self.progress_bar = progress_bar
        self.verbosity = verbosity
        self.__name = type(self).__name__
        self.__initialized = False
        self.__entered = False

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
            n_chunks_states = min(self.n_procs, n_states)
            chunk_size_states = max(int(n_states / self.n_procs), 1)
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
                    n_chunks_targets = min(self.n_procs, n_targets)
                    chunk_size_targets = max(int(n_targets / self.n_procs), 1)
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

    def combine_results(
        self,
        algo,
        futures,
        model_data,
        out_vars,
        out_coords,
        n_chunks_states,
        n_chunks_targets,
        goal_data,
        iterative,
        write_nc=None,
        results=None,
    ):
        """
        Combines chunk results into final Dataset

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        futures: dict
            The futures from the chunk calculations,
            key: (chunki_states, chunki_targets),
            value: future unpacking to (results, cstore)
        model_data: xarray.Dataset
            The initial model data
        out_vars: list of str
            Names of the output variables
        out_coords: list of str
            Names of the output coordinates
        n_chunks_states: int
            The number of states chunks
        n_chunks_targets: int
            The number of targets chunks
        goal_data: foxes.core.Data
            Either fdata or tdata
        iterative: bool
            Flag for use within the iterative algorithm
        write_nc: dict, optional
            Parameters for writing results to netCDF files, e.g.
            {'out_dir': 'results', 'base_name': 'calc_results', 
            'ret_data': False, 'split': 1000}.
            
            The split parameter controls how the output is split:
            - 'chunks': one file per chunk (fastest method),
            - 'input': split according to sizes of multiple states input files,
            - int: split with this many states per file,
            - None: create a single output file.

            Use ret_data = False together with non-single file writing
            to avoid constructing the full Dataset in memory.
        results: dict, options
            The evaluated futures

        Returns
        -------
        ds: xarray.Dataset, optional
            The final results dataset

        """
        assert (futures is not None and results is None) or (
            futures is None and results is not None
        ), f"{self.name}: Either futures or results must be provided, not both"

        self.print(f"{self.name}: Combining results", level=2)

        coords = {}
        vrb = max(self.verbosity - 1, 0)
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()

        # read parameters for file writing
        ret_data = True
        split_size = None
        write_on_fly = False
        write_from_ds = False
        if write_nc is not None:
            out_dir = get_output_path(write_nc.get("out_dir", "."))
            base_name = write_nc["base_name"]
            ret_data = write_nc.get("ret_data", False)
            split_mode = write_nc.get("split", None)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fpath = out_dir/(base_name + "_*.nc")
            if split_mode == "chunks":
                # files are already written during chunk calculations
                self.print(f"{self.name}: Wrote results to '{out_fpath}', using split = {split_mode}, ret_data = {ret_data}")
                if not ret_data:
                    return None
            elif split_mode == "input":
                gen_size = algo.states.gen_states_split_size()
                split_size = next(gen_size)
            elif isinstance(split_mode, int):
                split_size = split_mode
            elif split_mode is None:
                split_size = None
            else:
                raise ValueError(f"Invalid split mode '{split_mode}' in 'write_nc', expected 'chunks', 'input', int or None")
            if split_size is None:
                out_fpath = out_dir/(base_name + ".nc")
            if split_mode != "chunks":
                write_on_fly = ret_data == False and split_size is not None
                write_from_ds = not write_on_fly
                ret_data = write_nc.get("ret_data", write_from_ds)
                self.print(f"{self.name}: Results written to '{out_fpath}', using split = {split_mode}, on_fly = {write_on_fly}, ret_data = {ret_data}")

        def _red_dims(data_vars):
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
                    and goal_data.sizes[FC.TARGET] > n_chunks_targets
                    and d.shape[1:] == (n_chunks_targets, 1)
                ):
                    dvars[v] = ((FC.STATE,), d[:, 0, 0])
                else:
                    dvars[v] = (dims, d)
            return dvars

        fcounter = 0
        def _write_parts_on_fly(data_vars, scount, wcount):
            """Helper function for writing chunked results to netCDF,
                without ever constructing the full Dataset in memory
            """
            nonlocal fcounter, split_size
            wfutures = []
            if split_size is not None:
                splits = min(split_size, algo.n_states - wcount)
                while algo.n_states - wcount > 0 and scount - wcount >= splits:
                    for v in data_vars.keys():
                        if len(data_vars[v][1]) > 1:
                            data_vars[v][1] = np.concatenate(data_vars[v][1], axis=0)
                        elif len(data_vars[v][1]) == 1:
                            data_vars[v][1] = data_vars[v][1][0]
                    
                    dvars = {v: (d[0], d[1][:splits]) for v, d in data_vars.items()}
                    dvars = _red_dims(dvars)

                    crds = {v: d for v, d in coords.items()}
                    crds[FC.STATE] = coords[FC.STATE][wcount:wcount+splits]
                    if scount - wcount == splits:
                        for v in data_vars.keys():
                            data_vars[v][1] = []
                    else:
                        for v in data_vars.keys():
                            data_vars[v][1] = [data_vars[v][1][splits:]]

                    ds = Dataset(coords=crds,data_vars=dvars)
                    fpath = out_dir / f"{base_name}_{fcounter:04d}.nc"
                    future = self.submit(write_nc_file, ds, fpath, nc_engine=config.nc_engine, verbosity=vrb)
                    wfutures.append(future)
                    del ds, crds, dvars, future

                    wcount += splits
                    fcounter += 1

                    if algo.n_states - wcount > 0:
                        if split_mode == "input":
                            try:
                                split_size = next(gen_size)
                            except StopIteration:
                                split_size = algo.n_states - wcount
                        splits = min(split_size, algo.n_states - wcount)

            return wcount, wfutures

        data_vars = {}
        def _get_res_vars(result, data_vars):
            """Helper function for extracting results variables"""
            res_vars = list(result.keys())
            for v in out_vars:
                if v in res_vars:
                    data_vars[v] = [out_coords, []]
                else:
                    data_vars[v] = (goal_data[v].dims, goal_data[v].to_numpy())
            return res_vars

        keys = list(futures.keys()) if futures is not None else list(results.keys())
        if futures is not None:
            self.print(f"{self.name}: Computing {len(keys)} chunks using {self.n_procs} processes")
            vlevel = 0
        else:
            self.print(f"{self.name}: Combining results from {len(keys)} chunks")
            vlevel = 1
        pbar = None
        if self.verbosity > vlevel and self.has_progress_bar:
            pbar = tqdm(total=len(keys))

        scount = 0
        wcount = 0
        wfutures = []
        res_vars = None
        pdone = -1
        if n_chunks_targets == 1:
            for key in keys:
                if futures is not None:
                    r, cstore = self.await_result(futures.pop(key))
                else:
                    r, cstore = results.pop(key)
                if res_vars is None:
                    res_vars =_get_res_vars(r, data_vars)
                if iterative:
                    for k, c in cstore.items():
                        if k in algo.chunk_store:
                            algo.chunk_store[k].update(c)
                        else:
                            algo.chunk_store[k] = c
                scount += r[res_vars[0]].shape[0]
                for v in res_vars:
                    if v in data_vars:
                        data_vars[v][1].append(r[v])
                if write_on_fly:
                    wcount, ftrs = _write_parts_on_fly(data_vars, scount, wcount)
                    wfutures += ftrs
                    del ftrs
                del r, cstore
                if pbar is not None:
                    pbar.update()
                elif self.verbosity > vlevel and self.prints_progress and len(keys) > 1:
                    pr = int(100 * key[0]/(len(keys) - 1))
                    if pr > pdone:
                        pdone = pr
                        print(f"{self.name}: Completed {key[0]} of {len(keys)} chunks, {pdone}%")
        else:
            pdone = -1
            counter = 0
            for chunki_states in range(n_chunks_states):
                tres = None
                for chunki_points in range(n_chunks_targets):
                    found = False
                    for key in keys:
                        if key == (chunki_states, chunki_points):
                            if futures is not None:
                                r, cstore = self.await_result(futures.pop(key))
                            else:
                                r, cstore = results.pop(key)
                            if res_vars is None:
                                res_vars =_get_res_vars(r, data_vars)
                            if tres is None:
                                tres = {v: [] for v in res_vars}
                            
                            found = True
                            break
                    if not found:
                        raise KeyError(f"{self.name}: Missing future for key {key}")
                    
                    for v in res_vars:
                        tres[v].append(r[v])
                    if iterative:
                        for k, c in cstore.items():
                            if k in algo.chunk_store:
                                algo.chunk_store[k].update(c)
                            else:
                                algo.chunk_store[k] = c
                    del r, cstore

                    if pbar is not None:
                        pbar.update()
                    elif self.verbosity > vlevel and self.prints_progress and len(keys) > 1:
                        pr = int(100 * counter/(len(keys) - 1))
                        if pr > pdone:
                            pdone = pr
                            print(f"{self.name}: Completed {counter} of {len(keys)} chunks, {pdone}%")
                    counter += 1

                found = False
                for v in res_vars:
                    if v in data_vars:
                        data_vars[v][1].append(np.concatenate(tres[v], axis=1))
                        if not found:
                            scount += data_vars[v][1][-1].shape[0]
                        found = True
                del tres
                if write_on_fly:
                    wcount, ftrs = _write_parts_on_fly(data_vars, scount, wcount)
                    wfutures += ftrs
                    del ftrs
        for wf in wfutures:
            self.await_result(wf)
        del wfutures

        if pbar is not None:
            pbar.close()
        self.print(f"{self.name}: Completed all {len(keys)} chunks\n", level=vlevel+1)

        # if not iterative or algo.final_iteration:
        #    algo.reset_chunk_store()

        if ret_data or write_from_ds:
            for v in res_vars:
                if v in data_vars:
                    if len(data_vars[v][1]) > 1:
                        data_vars[v][1] = np.concatenate(data_vars[v][1], axis=0)
                    elif len(data_vars[v][1]) == 1:
                        data_vars[v][1] = data_vars[v][1][0]

            dvars = _red_dims(data_vars)
            ds = Dataset(
                coords=coords,
                data_vars=dvars,
            )

            if write_from_ds:
                if split_size is None:
                    write_nc_file(ds, out_fpath, nc_engine=config.nc_engine, verbosity=vrb)
                else:
                    wcount = 0
                    fcounter = 0
                    wfutures = []
                    while wcount < algo.n_states:
                        splits = min(split_size, algo.n_states - wcount)
                        dssub = ds.isel({FC.STATE: slice(wcount, wcount + splits)})

                        fpath = out_dir / f"{base_name}_{fcounter:04d}.nc"
                        future = self.submit(write_nc_file, dssub, fpath, nc_engine=config.nc_engine, verbosity=vrb)
                        wfutures.append(future)
                        del dssub, future

                        wcount += splits
                        fcounter += 1

                        if wcount < algo.n_states and split_mode == "input":
                            try:
                                split_size = next(gen_size)
                            except StopIteration:
                                split_size = algo.n_states - wcount

                    for wf in wfutures:
                        self.await_result(wf)

        if ret_data:
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
            xarray="XArrayEngine",
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
