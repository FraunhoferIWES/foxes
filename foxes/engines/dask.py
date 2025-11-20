import numpy as np
import xarray as xr
from copy import deepcopy
from tqdm import tqdm

from foxes.core import Engine, MData, FData, TData
from foxes.utils import import_module
import foxes.constants as FC

from .pool import _run_shared, _write_chunk_results, _write_ani

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


class DaskBaseEngine(Engine):
    """
    Abstract base class for foxes calculations with dask.

    Parameters
    ----------
    dask_config: dict
        The dask configuration parameters

    :group: engines

    """

    def __init__(
        self,
        *args,
        dask_config={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the base class
        dask_config: dict, optional
            The dask configuration parameters
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        load_dask()

        self.dask_config = dask_config
        self._dask_progress_bar = False
        self._pbar = None

    def __enter__(self):
        if self._dask_progress_bar:
            self._pbar = ProgressBar(minimum=2)
            self._pbar.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        if self._dask_progress_bar and self._pbar is not None:
            self._pbar.__exit__(*args)
        super().__exit__(*args)

    def initialize(self):
        """
        Initializes the engine.
        """
        dask.config.set(**self.dask_config)
        super().initialize()

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
        return delayed(f)(*args, **kwargs)

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
        return False

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
        return future.compute()

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

    def chunk_data(self, data):
        """
        Applies the selected chunking

        Parameters
        ----------
        data: xarray.Dataset
            The data to be chunked

        Returns
        -------
        data: xarray.Dataset
            The chunked data

        """
        cks = {}
        cks[FC.STATE] = min(data.sizes[FC.STATE], self.chunk_size_states)
        if FC.TARGET in data.sizes:
            cks[FC.TARGET] = min(data.sizes[FC.TARGET], self.chunk_size_points)

        if len(set(cks.keys()).intersection(data.coords.keys())):
            return data.chunk({v: d for v, d in cks.items() if v in data.coords})
        else:
            return data

    def finalize(self, *exit_args, **exit_kwargs):
        """
        Finalizes the engine.

        Parameters
        ----------
        exit_args: tuple, optional
            Arguments from the exit function
        exit_kwargs: dict, optional
            Arguments from the exit function

        """
        dask.config.refresh()
        super().finalize(*exit_args, **exit_kwargs)


class DaskEngine(DaskBaseEngine):
    """
    The dask engine for delayed foxes calculations.

    :group: engines

    """

    def __init__(
        self,
        *args,
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
        super().__init__(*args, progress_bar=None, **kwargs)
        self._dask_progress_bar = progress_bar

    def run_calculation(
        self,
        algo,
        model,
        model_data=None,
        farm_data=None,
        point_data=None,
        out_vars=[],
        chunk_store={},
        sel=None,
        isel=None,
        iterative=False,
        write_nc=None,
        write_chunk_ani=None,
        **calc_pars,
    ):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel
            The model that whose calculate function
            should be run
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset
            The initial farm data
        point_data: xarray.Dataset
            The initial point data
        out_vars: list of str, optional
            Names of the output variables
        chunk_store: foxes.utils.Dict
            The chunk store
        sel: dict, optional
            Selection of coordinate subsets
        isel: dict, optional
            Selection of coordinate subsets index values
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
        write_chunk_ani: dict, optional
            Parameters for writing chunk animations, e.g.
            {'fpath_base': 'results/chunk_animation', 'vars': ['WS'],
            'resolution': 100, 'chunk': 5}.'}
            The chunk is either an integer that refers to a states chunk,
            or a  tuple (states_chunk_index, points_chunk_index), or a list
            of such entries.
        calc_pars: dict, optional
            Additional parameters for the model.calculate()

        Returns
        -------
        results: xarray.Dataset
            The model results

        """
        # reset chunk store:
        algo.reset_chunk_store(chunk_store)

        # subset selection:
        model_data, farm_data, point_data = self.select_subsets(
            model_data, farm_data, point_data, sel=sel, isel=isel
        )

        # basic checks:
        super().run_calculation(algo, model, model_data, farm_data, point_data)

        # prepare:
        n_states = model_data.sizes[FC.STATE]
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = xr.Dataset()
        goal_data = farm_data if point_data is None else point_data
        algo.reset_chunk_store(chunk_store)

        # calculate chunk sizes:
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(
            n_states, n_targets
        )
        n_chunks_states = len(chunk_sizes_states)
        n_chunks_targets = len(chunk_sizes_targets)
        self.print(
            f"Selecting n_chunks_states = {n_chunks_states}, n_chunks_targets = {n_chunks_targets}",
            level=2,
        )

        # prepare and submit chunks:
        self.start_chunk_calculation(
            algo,
            coords=coords,
            goal_data=goal_data,
            n_chunks_states=n_chunks_states,
            n_chunks_targets=n_chunks_targets,
            iterative=iterative,
            write_nc=write_nc,
        )

        # submit chunks:
        n_chunks_all = n_chunks_states * n_chunks_targets
        self.print(
            f"Submitting {n_chunks_all} chunks to {self.n_workers} workers", level=2
        )
        pbar = (
            tqdm(total=n_chunks_all)
            if self.verbosity > 1 and self._dask_progress_bar
            else None
        )
        futures = {}
        i0_states = 0
        counter = 0
        pdone = -1
        for chunki_states in range(n_chunks_states):
            i1_states = i0_states + chunk_sizes_states[chunki_states]
            i0_targets = 0
            for chunki_points in range(n_chunks_targets):
                key = (chunki_states, chunki_points)
                i1_targets = i0_targets + chunk_sizes_targets[chunki_points]

                # get this chunk's data:
                data = self.get_chunk_input_data(
                    algo=algo,
                    model_data=model_data,
                    farm_data=farm_data,
                    point_data=point_data,
                    states_i0_i1=(i0_states, i1_states),
                    targets_i0_i1=(i0_targets, i1_targets),
                    out_vars=out_vars,
                    chunki_states=chunki_states,
                    chunki_points=chunki_points,
                    n_chunks_states=n_chunks_states,
                    n_chunks_points=n_chunks_targets,
                )

                # submit model calculation:
                futures[(chunki_states, chunki_points)] = delayed(_run_shared)(
                    algo,
                    model,
                    *data,
                    chunk_key=key,
                    out_coords=out_coords,
                    write_nc=write_nc,
                    write_chunk_ani=write_chunk_ani,
                    **calc_pars,
                )
                del data

                i0_targets = i1_targets

                counter += 1
                if pbar is not None:
                    pbar.update()
                elif self.verbosity > 1 and self.prints_progress:
                    pr = int(100 * counter / n_chunks_states)
                    if pr > pdone:
                        pdone = pr
                        print(
                            f"{self.name}: Submitted {counter} of {n_chunks_states} chunks, {pdone}%"
                        )
            i0_states = i1_states

        del farm_data, point_data, calc_pars
        if pbar is not None:
            pbar.close()

        # wait for results:
        if n_chunks_all > 1 or self.verbosity > 1:
            self.print(
                f"Computing {n_chunks_all} chunks using {self.n_workers} workers"
            )
        results = dask.compute(futures)[0]
        futures = None

        self.update_chunk_progress(
            algo,
            results=results,
            out_coords=out_coords,
            goal_data=goal_data,
            out_vars=out_vars,
            futures=futures,
        )

        return self.end_chunk_calculation(algo)


def _run_on_cluster(
    algo,
    model,
    *data,
    names,
    dims,
    out_coords,
    mdata_size,
    fdata_size,
    iterative,
    chunk_store,
    chunki_states,
    chunki_points,
    n_chunks_states,
    n_chunks_points,
    i0_states,
    write_nc,
    write_chunk_ani,
    cpars,
):
    """Helper function for running on a cluster"""

    algo.reset_chunk_store(chunk_store)

    mdata = MData(
        data={names[i]: data[i] for i in range(mdata_size)},
        dims={names[i]: dims[i] for i in range(mdata_size)},
        chunki_states=chunki_states,
        chunki_points=chunki_points,
        n_chunks_states=n_chunks_states,
        n_chunks_points=n_chunks_points,
        states_i0=i0_states,
    )

    fdata_end = mdata_size + fdata_size
    fdata = FData(
        data={names[i]: data[i].copy() for i in range(mdata_size, fdata_end)},
        dims={names[i]: dims[i] for i in range(mdata_size, fdata_end)},
        chunki_states=chunki_states,
        chunki_points=chunki_points,
        n_chunks_states=n_chunks_states,
        n_chunks_points=n_chunks_points,
        states_i0=i0_states,
    )

    tdata = None
    if len(data) > fdata_end:
        tdata = TData(
            data={names[i]: data[i].copy() for i in range(fdata_end, len(data))},
            dims={names[i]: dims[i] for i in range(fdata_end, len(data))},
            chunki_states=chunki_states,
            chunki_points=chunki_points,
            n_chunks_states=n_chunks_states,
            n_chunks_points=n_chunks_points,
            states_i0=i0_states,
        )

    data = [d for d in [mdata, fdata, tdata] if d is not None]

    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store() if iterative else {}

    k = (chunki_states, chunki_points)
    cstore = {k: chunk_store[k]} if k in chunk_store else {}

    _write_ani(algo, k, write_chunk_ani, *data)
    results = _write_chunk_results(algo, results, write_nc, out_coords, data[0])

    return results, cstore


class LocalClusterEngine(DaskBaseEngine):
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
        cluster_pars={},
        client_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the DaskBaseEngine class
        cluster_pars: dict
            Parameters for the cluster
        client_pars: dict
            Parameters for the client of the cluster
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        load_distributed()

        self.cluster_pars = cluster_pars
        self.client_pars = client_pars

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
        return super().__enter__()

    def __exit__(self, *args):
        self.print(f"Shutting down {type(self._cluster).__name__}")
        # self._client.retire_workers()
        # from time import sleep
        # sleep(1)
        # self._client.shutdown()
        self._client.__exit__(*args)
        self._cluster.__exit__(*args)
        super().__exit__(*args)

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

    def run_calculation(
        self,
        algo,
        model,
        model_data=None,
        farm_data=None,
        point_data=None,
        out_vars=[],
        chunk_store={},
        sel=None,
        isel=None,
        iterative=False,
        write_nc=None,
        write_chunk_ani=None,
        **calc_pars,
    ):
        """
        Runs the model calculation

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm object
        model: foxes.core.DataCalcModel
            The model that whose calculate function
            should be run
        model_data: xarray.Dataset
            The initial model data
        farm_data: xarray.Dataset
            The initial farm data
        point_data: xarray.Dataset
            The initial point data
        out_vars: list of str, optional
            Names of the output variables
        chunk_store: foxes.utils.Dict
            The chunk store
        sel: dict, optional
            Selection of coordinate subsets
        isel: dict, optional
            Selection of coordinate subsets index values
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
        write_chunk_ani: dict, optional
            Parameters for writing chunk animations, e.g.
            {'fpath_base': 'results/chunk_animation', 'vars': ['WS'],
            'resolution': 100, 'chunk': 5}.'}
            The chunk is either an integer that refers to a states chunk,
            or a  tuple (states_chunk_index, points_chunk_index), or a list
            of such entries.
        calc_pars: dict, optional
            Additional parameters for the model.calculate()

        Returns
        -------
        results: xarray.Dataset
            The model results

        """
        # subset selection:
        model_data, farm_data, point_data = self.select_subsets(
            model_data, farm_data, point_data, sel=sel, isel=isel
        )

        # basic checks:
        super().run_calculation(algo, model, model_data, farm_data, point_data)

        # prepare:
        n_states = model_data.sizes[FC.STATE]
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = xr.Dataset()
        goal_data = farm_data if point_data is None else point_data

        # calculate chunk sizes:
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(
            n_states, n_targets
        )
        n_chunks_states = len(chunk_sizes_states)
        n_chunks_targets = len(chunk_sizes_targets)
        self.print(
            f"Selecting n_chunks_states = {n_chunks_states}, n_chunks_targets = {n_chunks_targets}",
            level=2,
        )

        # scatter algo and model:
        falgo = self._client.scatter(algo, broadcast=True)
        fmodel = self._client.scatter(model, broadcast=True)
        cpars = self._client.scatter(calc_pars, broadcast=True)
        all_data = [falgo, fmodel, cpars]

        # prepare chunks:
        self.start_chunk_calculation(
            algo,
            coords=coords,
            goal_data=goal_data,
            n_chunks_states=n_chunks_states,
            n_chunks_targets=n_chunks_targets,
            iterative=iterative,
            write_nc=write_nc,
        )

        # submit chunks:
        futures = {}
        results = {}
        i0_states = 0
        for chunki_states in range(n_chunks_states):
            i1_states = i0_states + chunk_sizes_states[chunki_states]
            i0_targets = 0
            for chunki_points in range(n_chunks_targets):
                i1_targets = i0_targets + chunk_sizes_targets[chunki_points]

                # get this chunk's data:
                data = self.get_chunk_input_data(
                    algo=algo,
                    model_data=model_data,
                    farm_data=farm_data,
                    point_data=point_data,
                    states_i0_i1=(i0_states, i1_states),
                    targets_i0_i1=(i0_targets, i1_targets),
                    out_vars=out_vars,
                    chunki_states=chunki_states,
                    chunki_points=chunki_points,
                    n_chunks_states=n_chunks_states,
                    n_chunks_points=n_chunks_targets,
                )

                # scatter data:
                fut_data = []
                names = []
                dims = []
                ldims = [d.loop_dims for d in data]
                for dt in data:
                    for k, d in dt.items():
                        fut_data.append(self._client.scatter(d, hash=False))
                        names.append(k)
                        dims.append(dt.dims[k])
                names = self._client.scatter(names)
                dims = self._client.scatter(dims)
                ldims = self._client.scatter(ldims)
                all_data += [fut_data, names, dims, ldims]

                # scatter chunk store data:
                cstore = chunk_store
                if len(cstore):
                    cstore = self._client.scatter(cstore, hash=False)
                    all_data.append(cstore)

                # submit model calculation:
                futures[(chunki_states, chunki_points)] = self.submit(
                    _run_on_cluster,
                    falgo,
                    fmodel,
                    *fut_data,
                    names=names,
                    dims=dims,
                    out_coords=out_coords,
                    mdata_size=len(data[0]),
                    fdata_size=len(data[1]),
                    iterative=iterative,
                    chunk_store=cstore,
                    chunki_states=chunki_states,
                    chunki_points=chunki_points,
                    n_chunks_states=n_chunks_states,
                    n_chunks_points=n_chunks_targets,
                    i0_states=i0_states,
                    write_nc=write_nc,
                    write_chunk_ani=write_chunk_ani,
                    cpars=cpars,
                    retries=10,
                )
                del fut_data, cstore

                i0_targets = i1_targets

                while len(futures) > self.n_workers * 2:
                    k = next(iter(futures))
                    results[k] = self.await_result(futures.pop(k))

                    self.update_chunk_progress(
                        algo,
                        results=results,
                        out_coords=out_coords,
                        goal_data=goal_data,
                        out_vars=out_vars,
                        futures=futures,
                    )

            i0_states = i1_states

        del falgo, fmodel, farm_data, point_data, calc_pars

        for k in list(futures.keys()):
            results[k] = self.await_result(futures.pop(k))

            self.update_chunk_progress(
                algo,
                results=results,
                out_coords=out_coords,
                goal_data=goal_data,
                out_vars=out_vars,
                futures=futures,
            )

        return self.end_chunk_calculation(algo)


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

        return DaskBaseEngine.__enter__(self)
