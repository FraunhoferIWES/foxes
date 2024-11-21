import numpy as np
import xarray as xr
from copy import deepcopy
from tqdm import tqdm

from foxes.core import Engine, MData, FData, TData
from foxes.utils import import_module
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

dask = None
distributed = None


def delayed(func):
    """A dummy decorator"""
    return func


def load_dask():
    """On-demand loading of the dask package"""
    global dask, ProgressBar, delayed
    if dask is None:
        dask = import_module("dask", hint="pip install dask")
        ProgressBar = import_module(
            "dask.diagnostics", hint="pip install dask"
        ).ProgressBar
        delayed = dask.delayed


def load_distributed():
    """On-demand loading of the distributed package"""
    global distributed
    if distributed is None:
        distributed = import_module("distributed", hint="pip install distributed")


class DaskBaseEngine(Engine):
    """
    Abstract base class for foxes calculations with dask.

    Parameters
    ----------
    dask_config: dict
        The dask configuration parameters
    progress_bar: bool
        Flag for showing progress bar

    :group: engines

    """

    def __init__(
        self,
        *args,
        dask_config={},
        progress_bar=True,
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
        progress_bar: bool
            Flag for showing progress bar
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        load_dask()

        self.dask_config = dask_config
        self.progress_bar = progress_bar

    def __enter__(self):
        if self.progress_bar:
            self._pbar = ProgressBar(minimum=2)
            self._pbar.__enter__()
        return super().__enter__()

    def __exit__(self, *args):
        if self.progress_bar:
            self._pbar.__exit__(*args)
        super().__exit__(*args)

    def initialize(self):
        """
        Initializes the engine.
        """
        dask.config.set(**self.dask_config)
        super().initialize()

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


def _run_as_ufunc(
    state_inds,
    *ldata,
    algo,
    dvars,
    lvars,
    ldims,
    evars,
    edims,
    edata,
    loop_dims,
    out_vars,
    out_coords,
    calc_pars,
    init_vars,
    ensure_variables,
    calculate,
):
    """
    Wrapper that mitigates between apply_ufunc and `calculate`.
    """
    n_prev = len(init_vars)
    if n_prev:
        prev = ldata[:n_prev]
        ldata = ldata[n_prev:]

    # reconstruct original data:
    data = []
    for i, hvars in enumerate(dvars):
        v2l = {v: lvars.index(v) for v in hvars if v in lvars}
        v2e = {v: evars.index(v) for v in hvars if v in evars}

        hdata = {v: ldata[v2l[v]] if v in v2l else edata[v2e[v]] for v in hvars}
        hdims = {v: ldims[v2l[v]] if v in v2l else edims[v2e[v]] for v in hvars}

        if i == 0:
            data.append(
                MData(
                    data=hdata, dims=hdims, loop_dims=loop_dims, states_i0=state_inds[0]
                )
            )
        elif i == 1:
            data.append(
                FData(
                    data=hdata, dims=hdims, loop_dims=loop_dims, states_i0=state_inds[0]
                )
            )
        elif i == 2:
            data.append(
                TData(
                    data=hdata, dims=hdims, loop_dims=loop_dims, states_i0=state_inds[0]
                )
            )
        else:
            raise NotImplementedError(
                f"Not more than 3 data sets implemented, found {len(dvars)}"
            )

        del hdata, hdims, v2l, v2e

    # deduce output shape:
    oshape = []
    for li, l in enumerate(out_coords):
        for i, dims in enumerate(ldims):
            if l in dims:
                oshape.append(ldata[i].shape[dims.index(l)])
                break
        if len(oshape) != li + 1:
            raise ValueError(f"Failed to find loop dimension")

    # add zero output data arrays:
    odims = {v: tuple(out_coords) for v in out_vars}
    odata = {
        v: (
            np.full(oshape, np.nan, dtype=config.dtype_double)
            if v not in init_vars
            else prev[init_vars.index(v)].copy()
        )
        for v in out_vars
        if v not in data[-1]
    }

    if len(data) == 1:
        data.append(FData(odata, odims, loop_dims, states_i0=state_inds[0]))
    else:
        odata.update(data[-1])
        odims.update(data[-1].dims)
        if len(data) == 2:
            data[-1] = FData(odata, odims, loop_dims, states_i0=state_inds[0])
        else:
            data[-1] = TData(odata, odims, loop_dims, states_i0=state_inds[0])
    del odims, odata

    # link chunk state indices from mdata to fdata and tdata:
    if FC.STATE in data[0]:
        for d in data[1:]:
            d[FC.STATE] = data[0][FC.STATE]

    # link weights from mdata to fdata:
    if FV.WEIGHT in data[0]:
        data[1][FV.WEIGHT] = data[0][FV.WEIGHT]
        data[1].dims[FV.WEIGHT] = data[0].dims[FV.WEIGHT]

    # run model calculation:
    ensure_variables(algo, *data)
    results = calculate(algo, *data, **calc_pars)

    # replace missing results by first input data with matching shape:
    missing = set(out_vars).difference(results.keys())
    if len(missing):
        found = set()
        for v in missing:
            for dta in data:
                if v in dta and dta[v].shape == tuple(oshape):
                    results[v] = dta[v]
                    found.add(v)
                    break
        missing -= found
        if len(missing):
            raise ValueError(
                f"Missing results {list(missing)}, expected shape {oshape}"
            )
    del data

    # create output:
    n_vars = len(out_vars)
    data = np.zeros(oshape + [n_vars], dtype=config.dtype_double)
    for v in out_vars:
        data[..., out_vars.index(v)] = results[v]

    return data


class XArrayEngine(DaskBaseEngine):
    """
    The engine for foxes calculations via xarray.apply_ufunc.

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
        persist=True,
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
        persist: bool
            Flag for persisting xarray Dataset objects
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

        # find chunk sizes, if not given:
        chunk_size_states0 = self.chunk_size_states
        chunk_size_points0 = self.chunk_size_points
        n_states = model_data.sizes[FC.STATE]
        n_targets = point_data.sizes[FC.TARGET] if point_data is not None else 0
        chunk_sizes_states, chunk_sizes_targets = self.calc_chunk_sizes(
            n_states, n_targets
        )
        self.chunk_size_states = np.min(chunk_sizes_states)
        self.chunk_size_points = np.min(chunk_sizes_targets)
        self.print(
            f"{type(self).__name__}: Selecting chunk_size_states = {self.chunk_size_states}, chunk_size_points = {self.chunk_size_points}"
        )  # , level=2)

        # prepare:
        algo.reset_chunk_store(chunk_store)
        out_coords = model.output_coords()
        loop_dims = [d for d in self.loop_dims if d in out_coords]
        loopd = set(loop_dims)

        # extract loop-var dependent and independent data:
        ldata = []
        lvars = []
        ldims = []
        edata = []
        evars = []
        edims = []
        dvars = []
        ivars = []
        idims = []
        data = [
            self.chunk_data(d)
            for d in [model_data, farm_data, point_data]
            if d is not None
        ]
        for ds in data:

            hvarsl = [v for v, d in ds.items() if len(loopd.intersection(d.dims))]
            ldata += [ds[v] for v in hvarsl]
            ldims += [ds[v].dims for v in hvarsl]
            lvars += hvarsl

            hvarse = [v for v in ds.keys() if v not in hvarsl]
            edata += [ds[v].values for v in hvarse]
            edims += [ds[v].dims for v in hvarse]
            evars += hvarse

            for c, d in ds.coords.items():
                if c in loopd:
                    ldata.append(
                        self.chunk_data(
                            xr.DataArray(data=d.values, coords={c: d}, dims=[c])
                        )
                    )
                    ldims.append((c,))
                    lvars.append(c)
                else:
                    edata.append(d.values)
                    edims.append((c,))
                    evars.append(c)

            dvars.append(list(ds.keys()) + list(ds.coords.keys()))

        # apply persist:
        if persist:
            ldata = [d.persist() for d in ldata]

        # setup dask options:
        dargs = dict(output_sizes={FC.VARS: len(out_vars)})
        out_core_vars = [d for d in out_coords if d not in loop_dims] + [FC.VARS]
        if FC.TURBINE in loopd and FC.TURBINE not in ldims.values():
            dargs["output_sizes"][FC.TURBINE] = algo.n_turbines

        # find states_i0:
        state_inds = self.chunk_data(
            xr.DataArray(
                np.arange(ldata[0].sizes[FC.STATE]),
                dims=FC.STATE,
                coords={FC.STATE: ldata[0][FC.STATE].to_numpy()},
            )
        )

        # setup arguments for wrapper function:
        out_coords = loop_dims + list(set(out_core_vars).difference([FC.VARS]))
        wargs = dict(
            algo=algo,
            dvars=dvars,
            lvars=lvars,
            ldims=ldims,
            evars=evars,
            edims=edims,
            edata=edata,
            loop_dims=loop_dims,
            out_vars=out_vars,
            out_coords=out_coords,
            calc_pars=calc_pars,
            init_vars=ivars,
            ensure_variables=model.ensure_variables,
            calculate=model.calculate,
        )

        # run parallel computation:
        iidims = [[c for c in d if c not in loopd] for d in idims]
        icdims = [[c for c in d if c not in loopd] for d in ldims]
        results = xr.apply_ufunc(
            _run_as_ufunc,
            state_inds,
            *ldata,
            input_core_dims=[[]] + iidims + icdims,
            output_core_dims=[out_core_vars],
            output_dtypes=[config.dtype_double],
            dask="parallelized",
            dask_gufunc_kwargs=dargs,
            kwargs=wargs,
        )

        results = results.assign_coords({FC.VARS: out_vars}).to_dataset(dim=FC.VARS)

        # reset:
        self.chunk_size_states = chunk_size_states0
        self.chunk_size_points = chunk_size_points0

        # update data by calculation results:
        return results.compute(num_workers=self.n_procs)


@delayed
def _run_lazy(algo, model, iterative, chunk_store, i0_t0, *data, **cpars):
    """Helper function for lazy running"""
    algo.reset_chunk_store(chunk_store)
    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store() if iterative else {}
    cstore = {i0_t0: chunk_store[i0_t0]} if i0_t0 in chunk_store else {}
    return results, cstore


class DaskEngine(DaskBaseEngine):
    """
    The dask engine for delayed foxes calculations.

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

        # submit chunks:
        n_chunks_all = n_chunks_states * n_chunks_targets
        self.print(
            f"Submitting {n_chunks_all} chunks to {self.n_procs} processes", level=2
        )
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 1 else None
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
                )

                # submit model calculation:
                results[(chunki_states, chunki_points)] = _run_lazy(
                    deepcopy(algo),
                    deepcopy(model),
                    iterative,
                    chunk_store,
                    (i0_states, i0_targets),
                    *data,
                    **calc_pars,
                )
                del data

                i0_targets = i1_targets

                if pbar is not None:
                    pbar.update()

            i0_states = i1_states

        del farm_data, point_data, calc_pars
        if pbar is not None:
            pbar.close()

        # wait for results:
        if n_chunks_all > 1 or self.verbosity > 1:
            self.print(
                f"Computing {n_chunks_all} chunks using {self.n_procs} processes"
            )
        results = dask.compute(results)[0]

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


def _run_on_cluster(
    algo,
    model,
    *data,
    names,
    dims,
    mdata_size,
    fdata_size,
    loop_dims,
    iterative,
    chunk_store,
    i0_states,
    i0_targets,
    cpars,
):
    """Helper function for running on a cluster"""

    algo.reset_chunk_store(chunk_store)

    mdata = MData(
        data={names[i]: data[i] for i in range(mdata_size)},
        dims={names[i]: dims[i] for i in range(mdata_size)},
        loop_dims=loop_dims[0],
        states_i0=i0_states,
    )

    fdata_end = mdata_size + fdata_size
    fdata = FData(
        data={names[i]: data[i].copy() for i in range(mdata_size, fdata_end)},
        dims={names[i]: dims[i] for i in range(mdata_size, fdata_end)},
        loop_dims=loop_dims[1],
        states_i0=i0_states,
    )

    tdata = None
    if len(data) > fdata_end:
        tdata = TData(
            data={names[i]: data[i].copy() for i in range(fdata_end, len(data))},
            dims={names[i]: dims[i] for i in range(fdata_end, len(data))},
            loop_dims=loop_dims[2],
            states_i0=i0_states,
        )

    data = [d for d in [mdata, fdata, tdata] if d is not None]

    results = model.calculate(algo, *data, **cpars)
    chunk_store = algo.reset_chunk_store() if iterative else {}

    k = (i0_states, i0_targets)
    cstore = {k: chunk_store[k]} if k in chunk_store else {}
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
            n_workers=self.n_procs, **self.cluster_pars
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
        n_chunks_all = n_chunks_states * n_chunks_targets
        falgo = self._client.scatter(algo, broadcast=True)
        fmodel = self._client.scatter(model, broadcast=True)
        cpars = self._client.scatter(calc_pars, broadcast=True)
        all_data = [falgo, fmodel, cpars]

        # submit chunks:
        self.print(f"Submitting {n_chunks_all} chunks to {self.n_procs} processes")
        pbar = tqdm(total=n_chunks_all) if self.verbosity > 0 else None
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
                jobs[(chunki_states, chunki_points)] = self._client.submit(
                    _run_on_cluster,
                    falgo,
                    fmodel,
                    *fut_data,
                    names=names,
                    dims=dims,
                    mdata_size=len(data[0]),
                    fdata_size=len(data[1]),
                    loop_dims=ldims,
                    iterative=iterative,
                    chunk_store=cstore,
                    i0_states=i0_states,
                    i0_targets=i0_targets,
                    cpars=cpars,
                    retries=10,
                )
                del fut_data, cstore

                i0_targets = i1_targets

                if pbar is not None:
                    pbar.update()

            i0_states = i1_states

        del falgo, fmodel, farm_data, point_data, calc_pars
        if pbar is not None:
            pbar.close()

        # wait for results:
        self.print(f"Computing {n_chunks_all} chunks using {self.n_procs} processes")
        pbar = (
            tqdm(total=n_chunks_all)
            if n_chunks_all > 1 and self.verbosity > 0
            else None
        )
        results = {}
        for chunki_states in range(n_chunks_states):
            for chunki_points in range(n_chunks_targets):
                key = (chunki_states, chunki_points)
                results[key] = jobs.get(key).result()
                if pbar is not None:
                    pbar.update()
        if pbar is not None:
            pbar.close()

        results = self.combine_results(
            algo=algo,
            results=results,
            model_data=model_data,
            out_vars=out_vars,
            out_coords=out_coords,
            n_chunks_states=n_chunks_states,
            n_chunks_targets=n_chunks_targets,
            goal_data=goal_data,
            iterative=iterative,
        ).persist()

        # self._client.cancel(all_data)

        return results


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
            "dask_jobqueue", hint="pip install setuptools dask-jobqueue"
        )
        self._cluster = dask_jobqueue.SLURMCluster(**cargs)
        self._cluster.scale(jobs=nodes)
        self._cluster = self._cluster.__enter__()
        self._client = distributed.Client(self._cluster, **self.client_pars).__enter__()

        self.print(self._cluster)
        self.print(f"Dashboard: {self._client.dashboard_link}\n")
        print(self._cluster.job_script())

        return DaskBaseEngine.__enter__(self)
