from __future__ import annotations

from xarray import Dataset
from typing import Any

from foxes.core import Engine, EngineRunner
import foxes.constants as FC


class SingleChunkEngineRunner(EngineRunner):
    """
    Engine runner for SingleChunkEngine.

    :group: engines

    """

    def run(
        self,
        algo: Any,
        model: Any,
        mdata: Any,
        *data: Any,
        shared: Any,
        chunk_key: Any,
        out_dims: tuple[str, ...],
        write_nc: dict[str, Any] | None,
        write_chunk_ani: dict[str, Any] | None,
        **cpars: Any,
    ) -> tuple[dict[str, Any], dict[Any, Any]]:
        """Helper function for running in a single chunk."""
        if shared is not None:
            mdata.recombine_with_shared(shared)
        results = model.calculate(algo, mdata, *data, **cpars)
        cstore = (
            {chunk_key: algo.chunk_store[chunk_key]}
            if chunk_key in algo.chunk_store
            else {}
        )
        self._write_ani(algo, chunk_key, write_chunk_ani, mdata, *data)
        results = self._write_chunk_results(algo, results, write_nc, out_dims, mdata)
        if results is None:
            results = {}

        return results, cstore


class SingleChunkEngine(Engine):
    """
    Runs computations in a single chunk.

    :group: engines

    """

    def __init__(
        self,
        chunk_size_states: int | None = None,
        chunk_size_points: int | None = None,
        n_procs: int = 1,
        progress_bar: bool | None = True,
        verbosity: int = 1,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        chunk_size_states: int, optional
            Ignored for single chunk engine
        chunk_size_points: int, optional
            Ignored for single chunk engine
        n_procs: int
            Ignored for single chunk engine
        progress_bar: bool, optional
            Progress display mode
        verbosity: int
            Verbosity level

        """
        ignr = {
            "chunk_size_states": chunk_size_states,
            "chunk_size_points": chunk_size_points,
            "n_procs": n_procs,
        }
        for k, v in ignr.items():
            if v is not None and k != "n_procs" and verbosity > 1:
                print(f"{type(self).__name__}: Ignoring {k}")
            elif k == "n_procs" and v != 1 and verbosity > 1:
                print(f"{type(self).__name__}: Ignoring {k}")
        super().__init__(
            chunk_size_states=None,
            chunk_size_points=None,
            n_procs=1,
            progress_bar=progress_bar,
            verbosity=verbosity,
        )
        self.progress_bar = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def new_runner(self) -> SingleChunkEngineRunner:
        """
        Creates a new EngineRunner for running calculations in this engine

        Returns
        -------
        runner: foxes.core.EngineRunner
            The engine runner

        """
        return SingleChunkEngineRunner()

    def submit(self, f: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
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

        -------
        future: object
            The future object

        """
        return {"f": f, "args": args, "kwargs": kwargs, "result": None, "done": False}

    def await_result(self, future: dict[str, Any]) -> Any:
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
        if not future["done"]:
            f, args, kwargs = future.pop("f"), future.pop("args"), future.pop("kwargs")
            future["result"] = f(*args, **kwargs)
            future["done"] = True

        return future["result"]

    def future_is_done(self, future: dict[str, Any]) -> bool:
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
        return future["done"]

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
        return [func(input, *args, **kwargs) for input in inputs]

    def run_calculation(
        self,
        algo: Any,
        model: Any,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
        out_vars: list[str] | None = None,
        chunk_store: dict[Any, Any] | None = None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        iterative: bool = False,
        write_nc: dict[str, Any] | None = None,
        write_chunk_ani: dict[str, Any] | None = None,
        **calc_pars: Any,
    ) -> Dataset:
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
        if model_data is None:
            raise ValueError(f"{type(self).__name__}: model_data must not be None")
        if extra_data is None:
            extra_data = {}
        if out_vars is None:
            out_vars = []
        if chunk_store is None:
            chunk_store = {}

        # subset selection:
        (model_data, farm_data, point_data), n_states = self.select_subsets(
            model_data,
            farm_data,
            point_data,
            sel=sel,
            isel=isel,
            default_n_states=algo.n_states,
        )

        # basic checks:
        super().run_calculation(algo, model, model_data, farm_data, point_data)

        # prepare:
        algo.reset_chunk_store(chunk_store)
        n_states_eff = n_states if n_states is not None else algo.n_states
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        out_dims = model.output_coords()
        coords = {}
        if FC.STATE in out_dims and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        if farm_data is None:
            farm_data = Dataset()
        goal_data = farm_data if point_data is None else point_data

        # start calculation:
        with self.new_chunk_results_manager(
            algo,
            chunk_store=chunk_store,
            goal_data=goal_data,
            n_chunks_states=1,
            n_chunks_targets=1,
            out_vars=out_vars,
            out_dims=out_dims,
            coords=coords,
            iterative=iterative,
            write_nc=write_nc,
        ) as results_mgr:
            runner = self.new_runner()
            data = self.get_chunk_input_data(
                algo=algo,
                model_data=model_data,
                farm_data=farm_data,
                point_data=point_data,
                states_i0_i1=(0, n_states_eff),
                targets_i0_i1=(0, n_targets),
                out_vars=out_vars,
                chunki_states=0,
                chunki_points=0,
                n_chunks_states=1,
                n_chunks_points=1,
            )

            if len(extra_data) > 0:
                data[0].extra_data.update(extra_data)

            shared = None
            results, cstore = runner.run(
                algo,
                model,
                *data,
                shared=shared,
                chunk_key=(0, 0),
                out_dims=out_dims,
                write_nc=write_nc,
                write_chunk_ani=write_chunk_ani,
                **calc_pars,
            )
            chunk_results = {(0, 0): (results, cstore)}
            results_mgr.update(chunk_results)

            del (
                data,
                shared,
                results,
                cstore,
                chunk_results,
                farm_data,
                point_data,
                calc_pars,
            )

        return results_mgr.results
