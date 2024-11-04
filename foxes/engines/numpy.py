from tqdm import tqdm
from xarray import Dataset
from tqdm import tqdm

from foxes.core import Engine
import foxes.constants as FC

from .pool import _run


class NumpyEngine(Engine):
    """
    The numpy engine for foxes calculations.

    :group: engines

    """

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
        n_chunks_all = n_chunks_states * n_chunks_targets
        self.print(f"{type(self).__name__}: Looping over {n_chunks_all} chunks")
        pbar = (
            tqdm(total=n_chunks_all)
            if self.verbosity > 0 and n_chunks_all > 1
            else None
        )
        results = {}
        i0_states = 0
        for chunki_states in range(n_chunks_states):
            i1_states = i0_states + chunk_sizes_states[chunki_states]
            i0_targets = 0
            for chunki_points in range(n_chunks_targets):
                i1_targets = i0_targets + chunk_sizes_targets[chunki_points]

                i = chunki_states * n_chunks_targets + chunki_points

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
                key = (chunki_states, chunki_points)
                results[key] = _run(
                    algo,
                    model,
                    data,
                    iterative,
                    chunk_store,
                    (i0_states, i0_targets),
                    **calc_pars,
                )
                chunk_store.update(results[key][1])
                del data

                i0_targets = i1_targets

                if pbar is not None:
                    pbar.update()

            i0_states = i1_states

        del calc_pars, farm_data, point_data
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
