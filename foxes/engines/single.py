from xarray import Dataset

from foxes.core import Engine
import foxes.constants as FC

from .pool import _write_chunk_results


class SingleChunkEngine(Engine):
    """
    Runs computations in a single chunk.

    :group: engines

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for the base class
        kwargs: dict, optional
            Additional parameters for the base class

        """
        ignr = ["chunk_size_states", "chunk_size_points", "n_procs"]
        for k in ignr:
            if kwargs.pop(k, None) is not None and kwargs.get("verbosity", 1) > 1:
                print(f"{type(self).__name__}: Ignoring {k}")
        super().__init__(
            *args,
            chunk_size_states=None,
            chunk_size_points=None,
            n_procs=1,
            **kwargs,
        )
        self.progress_bar = None

    def __repr__(self):
        return f"{type(self).__name__}()"

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
        return f(*args, **kwargs)

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
        return future

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
        return [func(input, *args, **kwargs) for input in inputs]

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
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        out_coords = model.output_coords()
        coords = {}
        if FC.STATE in out_coords and FC.STATE in model_data.coords:
            coords[FC.STATE] = model_data[FC.STATE].to_numpy()
        algo.reset_chunk_store(chunk_store)

        # calculate:

        if n_states > 1:
            self.print(
                f"{type(self).__name__}: Running single chunk calculation for {n_states} states"
            )

        data = self.get_chunk_input_data(
            algo=algo,
            model_data=model_data,
            farm_data=farm_data,
            point_data=point_data,
            states_i0_i1=(0, n_states),
            targets_i0_i1=(0, n_targets),
            chunki_states=0,
            chunki_points=0,
            out_vars=out_vars,
        )

        results = model.calculate(algo, *data, **calc_pars)
        results = _write_chunk_results(algo, results, write_nc, out_coords, data[0])
        results = {(0, 0): (results, algo.chunk_store)}
        del data

        if farm_data is None:
            farm_data = Dataset()
        goal_data = farm_data if point_data is None else point_data

        return self.combine_results(
            algo=algo,
            futures=None,
            results=results,
            model_data=model_data,
            out_vars=out_vars,
            out_coords=out_coords,
            n_chunks_states=1,
            n_chunks_targets=1,
            goal_data=goal_data,
            iterative=iterative,
            write_nc=write_nc,
        )
