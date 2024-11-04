import xarray as xr
from abc import abstractmethod
from tqdm import tqdm

from foxes.core import Engine
import foxes.constants as FC


def _run(algo, model, data, iterative, chunk_store, i0_t0, **cpars):
    """Helper function for running in a single process"""
    algo.reset_chunk_store(chunk_store)
    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store() if iterative else {}
    cstore = {i0_t0: chunk_store[i0_t0]} if i0_t0 in chunk_store else {}
    return results, cstore


class PoolEngine(Engine):
    """
    Abstract engine for pool type parallelizations.

    :group: engines

    """

    @abstractmethod
    def _create_pool(self):
        """Creates the pool"""
        pass

    @abstractmethod
    def _submit(self, f, *args, **kwargs):
        """
        Submits to the pool

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
    def _result(self, future):
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
    def _shutdown_pool(self):
        """Shuts down the pool"""
        pass

    def __enter__(self):
        self._create_pool()
        return super().__enter__()

    def __exit__(self, *exit_args):
        self._shutdown_pool()
        super().__exit__(*exit_args)

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
        n_chunks_all = n_chunks_states * n_chunks_targets
        self.print(
            f"{type(self).__name__}: Submitting {n_chunks_all} chunks to {self.n_procs} processes",
            level=2,
        )
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 1 else None
        jobs = {}
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
                )

                # submit model calculation:
                jobs[(chunki_states, chunki_points)] = self._submit(
                    _run,
                    algo,
                    model,
                    data,
                    iterative,
                    chunk_store,
                    (i0_states, i0_targets),
                    **calc_pars,
                )
                del data

                i0_targets = i1_targets

                if pbar is not None:
                    pbar.update()

            i0_states = i1_states

        del calc_pars, farm_data, point_data
        if pbar is not None:
            pbar.close()

        # wait for results:
        if n_chunks_all > 1 or self.verbosity > 1:
            self.print(
                f"{type(self).__name__}: Computing {n_chunks_all} chunks using {self.n_procs} processes"
            )
        pbar = (
            tqdm(total=n_chunks_all)
            if n_chunks_all > 1 and self.verbosity > 0
            else None
        )
        results = {}
        for chunki_states in range(n_chunks_states):
            for chunki_points in range(n_chunks_targets):
                key = (chunki_states, chunki_points)
                results[key] = self._result(jobs.pop((chunki_states, chunki_points)))
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()

        return self.combine_results(
            algo=algo,
            results=results,
            model_data=model_data,
            out_vars=out_vars,
            out_coords=out_coords,
            n_chunks_states=n_chunks_states,
            n_chunks_targets=n_chunks_targets,
            goal_data=goal_data,
            iterative=iterative,
        )
