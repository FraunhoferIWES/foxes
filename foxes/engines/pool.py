import numpy as np
from xarray import Dataset
from abc import abstractmethod

from foxes.config import config, get_output_path
from foxes.core import Engine
from foxes.utils import write_nc as write_nc_file
from foxes.output import write_chunk_ani_xy
import foxes.constants as FC


def _write_chunk_results(algo, results, write_nc, out_coords, mdata):
    """Helper function for optionally writing chunk results to netCDF file"""
    ret_data = True
    if write_nc is not None and write_nc["split"] == "chunks":
        ret_data = write_nc.get("ret_data", False)
        out_dir = get_output_path(write_nc.get("out_dir", "."))
        base_name = write_nc["base_name"]
        ret_data = write_nc.get("ret_data", False)
        out_dir.mkdir(parents=True, exist_ok=True)

        coords = {}
        if FC.STATE in out_coords and FC.STATE in mdata:
            coords[FC.STATE] = mdata[FC.STATE]

        dvars = {}
        for v, d in results.items():
            if (
                out_coords == (FC.STATE, FC.TURBINE)
                and d.shape[1] == 1
                and algo.n_turbines > 1
            ):
                dvars[v] = ((FC.STATE,), d[:, 0])
            else:
                dvars[v] = (out_coords, d)

        ds = Dataset(coords=coords, data_vars=dvars)

        i0 = mdata.chunki_states
        t0 = mdata.chunki_points
        vrb = max(algo.verbosity - 1, 0)
        if out_coords == (FC.STATE, FC.TURBINE):
            fpath = out_dir / f"{base_name}_{i0:06d}.nc"
        else:
            fpath = out_dir / f"{base_name}_{i0:06d}_{t0:06d}.nc"
        write_nc_file(ds, fpath, nc_engine=config.nc_engine, verbosity=vrb)

    return results if ret_data else None


def _write_ani(algo, chunk_key, write_chunk_ani, *data):
    """Helper function for optionally writing chunk flow animations to file"""
    if write_chunk_ani is not None:
        pars = write_chunk_ani.copy()
        chk = pars.pop("chunk")

        def _do_run(chk):
            if isinstance(chk, list):
                for c in chk:
                    if _do_run(c):
                        return True
                return False
            else:
                return (
                    chk == chunk_key if isinstance(chk, tuple) else chk == chunk_key[0]
                )

        if _do_run(chk):
            write_chunk_ani_xy(algo, *data, **pars)


def _run(
    algo,
    model,
    *data,
    iterative,
    chunk_store,
    chunk_key,
    out_coords,
    write_nc,
    write_chunk_ani=None,
    **cpars,
):
    """Helper function for running in a single process"""
    algo.reset_chunk_store(chunk_store.copy())
    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store() if iterative else {}
    cstore = {chunk_key: chunk_store[chunk_key]} if chunk_key in chunk_store else {}
    _write_ani(algo, chunk_key, write_chunk_ani, *data)
    results = _write_chunk_results(algo, results, write_nc, out_coords, data[0])
    return results, cstore


def _run_shared(
    algo,
    model,
    *data,
    chunk_key,
    out_coords,
    write_nc,
    write_chunk_ani=None,
    **cpars,
):
    """Helper function for running in a single process"""
    results = model.calculate(algo, *data, **cpars)
    cstore = (
        {chunk_key: algo.chunk_store[chunk_key]}
        if chunk_key in algo.chunk_store
        else {}
    )
    _write_ani(algo, chunk_key, write_chunk_ani, *data)
    results = _write_chunk_results(algo, results, write_nc, out_coords, data[0])
    return results, cstore


def _run_map(func, inputs, *args, **kwargs):
    """Helper function for running map func on proc"""
    return [func(x, *args, **kwargs) for x in inputs]


class PoolEngine(Engine):
    """
    Abstract engine for pool type parallelizations.

    Parameters
    ----------
    share_cstore: bool
        Share chunk store between chunks

    :group: engines

    """

    def __init__(self, *args, share_cstore=False, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        share_cstore: bool
            Share chunk store between chunks
        kwargs: dict, optional
            Additional arguments for the base class

        """
        super().__init__(*args, **kwargs)
        self.share_cstore = share_cstore

    @abstractmethod
    def _create_pool(self):
        """Creates the pool"""
        pass

    @abstractmethod
    def _shutdown_pool(self):
        """Shuts down the pool"""
        pass

    def __enter__(self):
        self._create_pool()
        return super().__enter__()

    def __exit__(self, *exit_args):
        self._shutdown_pool()
        super().__exit__(*exit_args)

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
                futures.append(self.submit(_run_map, func, subi, *args, **kwargs))
            results = []
            for f in futures:
                results += self.await_result(f)
            return results

    def run_calculation(
        self,
        algo,
        model,
        model_data,
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
        farm_data: xarray.Dataset, optional
            The initial farm data
        point_data: xarray.Dataset, optional
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
        if self.share_cstore:
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
            farm_data = Dataset()
        goal_data = farm_data if point_data is None else point_data

        # DEBUG objec mem sizes:
        # from foxes.utils import print_mem
        # for m in [algo] + model.models:
        #    print_mem(m, pre_str="MULTIP CHECKING LARGE DATA", min_csize=9999)

        # calculate chunk sizes:
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(
            n_states, n_targets
        )
        n_chunks_states = len(chunk_sizes_states)
        n_chunks_targets = len(chunk_sizes_targets)
        self.print(
            f"{type(self).__name__}: Selecting n_chunks_states = {n_chunks_states}, n_chunks_targets = {n_chunks_targets}",
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

        futures = {}
        results = {}
        i0_states = 0
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
                if self.share_cstore:
                    futures[(chunki_states, chunki_points)] = self.submit(
                        _run_shared,
                        algo,
                        model,
                        *data,
                        chunk_key=key,
                        out_coords=out_coords,
                        write_nc=write_nc,
                        write_chunk_ani=write_chunk_ani,
                        **calc_pars,
                    )
                else:
                    futures[(chunki_states, chunki_points)] = self.submit(
                        _run,
                        algo,
                        model,
                        *data,
                        iterative=iterative,
                        chunk_store=chunk_store,
                        chunk_key=key,
                        out_coords=out_coords,
                        write_nc=write_nc,
                        write_chunk_ani=write_chunk_ani,
                        **calc_pars,
                    )
                del data

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
