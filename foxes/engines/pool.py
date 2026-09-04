from __future__ import annotations

import numpy as np
from xarray import Dataset
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.core import Engine
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import Algorithm, DataCalcModel, MData


def _run_map(func: Any, inputs: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Helper function for running map func on proc"""
    return [func(x, *args, **kwargs) for x in inputs]


class PoolEngine(Engine):
    """
    Abstract engine for pool type parallelizations.

    Parameters
    ----------
    share_cstore
        Whether to share the chunk store between chunks.
    pool_args
        Arguments for the pool constructor
    supports_shared_data
        Flag for whether this engine supports shared data for chunk calculations.
    min_shared_array_bytes
        Minimum array size in bytes for placing model data into process
        shared storage. Arrays with ``nbytes`` less than or equal to this
        threshold are transferred inline to workers. Supported top-level
        ``extra_data`` values use their complete payload size.
    """

    def __init__(
        self,
        *args: Any,
        share_cstore: bool = False,
        pool_args: dict[str, Any] | None = None,
        supports_shared_data: bool = True,
        min_shared_array_bytes: int = 65536,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        args
            Arguments for the base class
        pool_args
            Arguments for the pool constructor
        share_cstore
            Whether to share the chunk store between chunks.
        supports_shared_data
            Flag for whether this engine supports shared data for chunk calculations.
        min_shared_array_bytes
            Minimum array size in bytes for placing model data into process
            shared storage. Arrays with ``nbytes`` less than or equal to this
            threshold are transferred inline to workers. Supported top-level
            ``extra_data`` values use their complete payload size.
        kwargs
            Additional arguments for the base class
        """
        super().__init__(*args, **kwargs)
        self._pool: Any = None
        self.share_cstore = share_cstore
        self.pool_args = {} if pool_args is None else pool_args
        self.supports_shared_data = supports_shared_data
        self.min_shared_array_bytes = int(min_shared_array_bytes)

    @abstractmethod
    def _create_pool(self) -> None:
        """Creates the pool"""
        pass

    @abstractmethod
    def _shutdown_pool(self) -> None:
        """Shuts down the pool"""
        pass

    def _print_shared_data(self, shared_mdata: Any, verbosity: int) -> None:
        """Print diagnostics for data that is actually backed by shared storage."""
        if (
            verbosity > 1
            and shared_mdata is not None
            and (len(shared_mdata) > 0 or len(shared_mdata.extra_data) > 0)
        ):
            print(f"\n\n{type(self).__name__} shared data:\n")
            print(shared_mdata)
            print()

    def __enter__(self) -> PoolEngine:
        self._create_pool()
        super().__enter__()
        return self

    def __exit__(self, *exit_args: Any) -> None:
        self._shutdown_pool()
        super().__exit__(*exit_args)

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
        func
            Function to be called on each file,
            func(input, *args, **kwargs) -> data
        inputs
            The input data list
        args
            Arguments for func
        kwargs
            Keyword arguments for func

        Returns
        -------
        results
            Results for the submitted inputs

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
        algo: Algorithm,
        model: DataCalcModel,
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
        algo
            The algorithm object
        model
            The model that whose calculate function
            should be run
        model_data
            The initial model data
        farm_data
            The initial farm data
        point_data
            The initial point data
        extra_data
            Additional non-array input data from the models
        out_vars
            Names of the output variables
        chunk_store
            The chunk store
        sel
            Selection of coordinate subsets
        isel
            Selection of coordinate subsets index values
        iterative
            Flag for use within the iterative algorithm
        write_nc
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
        write_chunk_ani
            Parameters for writing chunk animations, e.g.
            {'fpath_base': 'results/chunk_animation', 'vars': ['WS'],
            'resolution': 100, 'chunk': 5}.'}
            The chunk is either an integer that refers to a states chunk,
            a tuple (states_chunk_index, points_chunk_index), or a list
            of chunk indices.
        calc_pars
            Additional parameters for the model.calculate()

        Returns
        -------
        results
            The model results

        """

        # prepare data:
        out_vars = [] if out_vars is None else out_vars
        chunk_store = {} if chunk_store is None else chunk_store
        extra_data = {} if extra_data is None else extra_data

        # reset chunk store:
        if self.share_cstore:
            algo.reset_chunk_store(chunk_store)
            new_chunk_store = chunk_store
        else:
            new_chunk_store = {}

        # subset selection:
        (model_data, farm_data, point_data), n_states = self.select_subsets(
            model_data,
            farm_data,
            point_data,
            sel=sel,
            isel=isel,
            default_n_states=algo.n_states,
        )
        if model_data is None:
            raise ValueError(f"Engine '{self.name}': Missing model_data")
        if n_states is None:
            raise ValueError(f"Engine '{self.name}': Could not determine n_states")
        if farm_data is not None and point_data is None:
            extra_data[FC.PREV_FARM_RESULTS] = farm_data
            farm_data = None

        # basic checks:
        super().run_calculation(algo, model, model_data, farm_data, point_data)

        # prepare:
        out_dims = model.output_coords()
        coords = {}
        if FC.STATE in out_dims and FC.STATE in model_data.coords:
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

        # start calculation:
        runner = self.new_runner()
        shared_memory: list[Any] = []
        shared_handle = None
        try:
            with self.new_chunk_results_manager(
                algo,
                chunk_store=new_chunk_store,
                goal_data=goal_data,
                n_chunks_states=n_chunks_states,
                n_chunks_targets=n_chunks_targets,
                out_vars=out_vars,
                out_dims=out_dims,
                coords=coords,
                iterative=iterative,
                write_nc=write_nc,
            ) as results_mgr:
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
                        mdata_chunk = data[0]
                        fdata_chunk = data[1]
                        tdata_chunk = data[2] if len(data) > 2 else None

                        if len(extra_data) > 0:
                            mdata_chunk.extra_data.update(extra_data)

                        if self.supports_shared_data:
                            if shared_handle is None and (
                                len(mdata_chunk) > 0 or len(extra_data) > 0
                            ):
                                shared_mdata = mdata_chunk.pop_shared(
                                    min_shared_array_bytes=self.min_shared_array_bytes
                                )
                                shared_handle = self.init_shared_memory(
                                    shared_memory,
                                    mdata=mdata_chunk,
                                    shared_mdata=shared_mdata,
                                    verbosity=max(self.verbosity, algo.verbosity),
                                )
                                del shared_mdata
                            elif shared_handle is not None:
                                self.prepare_chunk_mdata_for_shared(
                                    mdata_chunk, shared_handle
                                )

                        """
                        # For debugging: Check memory usage of main process before submitting the chunk calculation
                        import psutil
                        print(psutil.Process().pid, f"{algo.name} SUBMITTING {key} MEMORY:", psutil.Process().memory_info().rss / 1024 ** 2, "MB")
                        """

                        """
                        # For debugging: Check object sizes in memory
                        import psutil
                        import objsize
                        print(psutil.Process().pid, f"{algo.name} OBJECT SIZES BEFORE SUBMIT:", key, {k: objsize.get_deep_size(v) / 1024 ** 2 for k, v in {"algo": algo, "chunk_store": chunk_store, "shared": shared}.items()}, "MB")
                        """

                        # submit model calculation:
                        utm_zone = config.utm_zone if config.utm_zone_set else None
                        futures[(chunki_states, chunki_points)] = self.submit(
                            runner.run,
                            algo,
                            model,
                            mdata_chunk,
                            fdata_chunk,
                            tdata_chunk,
                            shared=shared_handle,
                            chunk_store=chunk_store,
                            chunk_key=key,
                            out_dims=out_dims,
                            write_nc=write_nc,
                            write_chunk_ani=write_chunk_ani,
                            utm_zone=utm_zone,
                            **calc_pars,
                        )
                        del data, mdata_chunk, fdata_chunk, tdata_chunk

                        while len(futures) > self.n_workers * 3:
                            k = next(iter(futures))
                            results[k] = self.await_result(futures.pop(k))
                            results_mgr.update(results, list(futures.values()))

                        i0_targets = i1_targets
                    i0_states = i1_states

                fkeys = list(futures.keys())
                for k in fkeys:
                    results[k] = self.await_result(futures.pop(k))
                    results_mgr.update(results, list(futures.values()))

                del calc_pars, farm_data, results, futures
        finally:
            self.release_shared_memory(shared_memory, shared_handle)
            del shared_memory, shared_handle

        # update chunk store:
        chunk_store.update(new_chunk_store)
        algo.reset_chunk_store(chunk_store)

        allow_none_results = write_nc is not None and write_nc.get("ret_data") is False
        if results_mgr.results is None:
            if allow_none_results:
                return None
            raise RuntimeError(
                f"{type(self).__name__} did not produce calculation results"
            )
        return results_mgr.results

    def init_shared_memory(
        self,
        shared_memory: list[Any],
        mdata: MData,
        shared_mdata: Any,
        verbosity: int = 0,
    ) -> Any:
        """
        Sets the shared memory for the chunk calculation

        Parameters
        ----------
        shared_memory
            The shared memory object for the chunk calculation
        mdata
            The mdata to be used in the chunk calculation
        shared_mdata
            The shared mdata to be used in the chunk calculation
        verbosity
            The verbosity level, 0=silent

        Returns
        -------
        handle
            The handle for accessing the shared data

        """
        mdata.recombine_with_shared(shared_mdata)
        return None

    def prepare_chunk_mdata_for_shared(self, mdata: MData, shared_handle: Any) -> None:
        """Hook for engine-specific mdata adjustments before worker submission.

        Parameters
        ----------
        mdata
            The chunk model data that will be sent to workers.
        shared_handle
            The handle that describes shared data for worker recombination.

        """
        pass

    def _prepare_chunk_extra_data_for_shared(
        self, mdata: MData, shared_handle: Any
    ) -> None:
        """Remove extra data that the shared handle restores in workers."""
        for key in shared_handle.get("extra_data_keys", ()):
            mdata.extra_data.pop(key, None)

    def release_shared_memory(
        self, shared_memory: list[Any], shared_handle: Any
    ) -> None:
        """
        Releases the shared memory after the chunk calculation

        Parameters
        ----------
        shared_memory
            The shared memory object for the chunk calculation
        shared_handle
            The handle for accessing the shared data

        """
        pass
