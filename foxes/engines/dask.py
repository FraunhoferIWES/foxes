import dask
import numpy as np
import xarray as xr
from distributed import Client, LocalCluster, progress
from dask.diagnostics import ProgressBar
from copy import deepcopy

from foxes.core import Engine, MData, FData, TData
import foxes.variables as FV
import foxes.constants as FC

def _wrap_calc(
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
            data.append(MData(data=hdata, dims=hdims, loop_dims=loop_dims))
        elif i == 1:
            data.append(FData(data=hdata, dims=hdims, loop_dims=loop_dims))
        elif i == 2:
            data.append(TData(data=hdata, dims=hdims, loop_dims=loop_dims))
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
            np.full(oshape, np.nan, dtype=FC.DTYPE)
            if v not in init_vars
            else prev[init_vars.index(v)].copy()
        )
        for v in out_vars
        if v not in data[-1]
    }

    if len(data) == 1:
        data.append(FData(odata, odims, loop_dims))
    else:
        odata.update(data[-1])
        odims.update(data[-1].dims)
        if len(data) == 2:
            data[-1] = FData(odata, odims, loop_dims)
        else:
            data[-1] = TData(odata, odims, loop_dims)
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
    data = np.zeros(oshape + [n_vars], dtype=FC.DTYPE)
    for v in out_vars:
        data[..., out_vars.index(v)] = results[v]

    return data
    
class DaskEngine(Engine):
    """
    The dask engine for foxes calculations.
    
    Parameters
    ----------
    dask_config: dict
        The dask configuration parameters
    cluster: str
        The dask cluster choice: 'local' or 'slurm'
    cluster_pars: dict
        Parameters for the cluster
    client_pars: dict
        Parameters for the client of the cluster
    progress_bar: bool
        Flag for showing progress bar
            
    :group: engines
    
    """
    def __init__(
        self, 
        dask_config={},
        cluster=None, 
        cluster_pars={}, 
        client_pars={}, 
        progress_bar=True,
        **kwargs,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        dask_config: dict, optional
            The dask configuration parameters
        cluster: str, optional
            The dask cluster choice: 'local' or 'slurm'
        cluster_pars: dict
            Parameters for the cluster
        client_pars: dict
            Parameters for the client of the cluster
        progress_bar: bool
            Flag for showing progress bar
        kwargs: dict, optional
            Additional parameters for the base class
            
        """
        super().__init__(**kwargs)
        self.dask_config = dask_config
        self.cluster = cluster
        self.cluster_pars = cluster_pars
        self.client_pars = client_pars
        self.progress_bar = progress_bar
        self._cluster = None
        self._client = None

    def initialize(self):
        """
        Initializes the engine.
        """       
        if self.cluster == "local":         
            self.print("Launching local dask cluster..")

            self._cluster = LocalCluster(**self.cluster_pars)
            self._client = Client(self._cluster, **self.client_pars)
            self.dask_config["scheduler"] = "distributed"

            self.print(self._cluster)
            self.print(f"Dashboard: {self._client.dashboard_link}\n")
            
        elif self.cluster == "slurm":
            from dask_jobqueue import SLURMCluster

            self.print("Launching dask cluster on HPC using SLURM..")

            cargs = deepcopy(self.cluster_pars)
            nodes = cargs.pop("nodes", 1)
            self._cluster = SLURMCluster(**cargs)
            self._cluster.scale(jobs=nodes)
            self._client = Client(self._cluster, **self.client_pars)
            self.dask_config["scheduler"] = "distributed"

            self.print(self._cluster)
            self.print(f"Dashboard: {self._client.dashboard_link}\n")
        
        if self.progress_bar:
            self._pbar = ProgressBar()
            self._pbar.register()

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
        if self.chunk_size_states is not None:
            cks[FC.STATE] = self.chunk_size_states
        if self.chunk_size_points is not None:
            cks[FC.TARGET] = self.chunk_size_points
        if len(set(cks.keys()).intersection(data.coords.keys())):
            return data.chunk(cks)
        else:
            return data
    
    def run_calculation(
        self, 
        algo,
        model, 
        model_data=None, 
        farm_data=None, 
        point_data=None, 
        out_vars=[],
        sel=None,
        isel=None,
        persist=True,
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
        sel: dict, optional
            Selection of coordinate subsets
        isel: dict, optional
            Selection of coordinate subsets index values
        persist: bool
            Flag for persisting xarray Dataset objects
        calc_pars: dict, optional
            Additional parameters for the model.calculate()
        
        Returns
        -------
        results: xarray.Dataset
            The model results
            
        """
        super().run_calculation(model, model_data, farm_data,
                                point_data, **calc_pars)
        
        # prepare:
        out_coords = model.output_coords()
        loopd = set(self.loop_dims)

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
        data = [self.chunk_data(d) for d in [model_data, farm_data, point_data] if d is not None]
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
                    ldata.append(self.chunk_data(xr.DataArray(data=d.values, coords={c: d}, dims=[c])))
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
            
        # subset selection:
        if sel is not None:
            nldata = []
            for ds in ldata:
                s = {k: v for k, v in sel.items() if k in ds.coords}
                if len(s):
                    nldata.append(ds.sel(s))
            ldata = nldata
            del nldata
        if isel is not None:
            nldata = []
            for ds in ldata:
                s = {k: v for k, v in isel.items() if k in ds.coords}
                if len(s):
                    nldata.append(ds.isel(s))
            ldata = nldata
            del nldata

        # setup dask options:
        dargs = dict(output_sizes={FC.VARS: len(out_vars)})
        out_core_vars = [d for d in out_coords if d not in self.loop_dims] + [FC.VARS]
        if FC.TURBINE in loopd and FC.TURBINE not in ldims.values():
            dargs["output_sizes"][FC.TURBINE] = algo.n_turbines
            
        # setup arguments for wrapper function:
        out_coords = self.loop_dims + list(set(out_core_vars).difference([FC.VARS]))
        wargs = dict(
            algo=algo,
            dvars=dvars,
            lvars=lvars,
            ldims=ldims,
            evars=evars,
            edims=edims,
            edata=edata,
            loop_dims=self.loop_dims,
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
            _wrap_calc,
            *ldata,
            input_core_dims=iidims + icdims,
            output_core_dims=[out_core_vars],
            output_dtypes=[FC.DTYPE],
            dask="parallelized",
            dask_gufunc_kwargs=dargs,
            kwargs=wargs,
        )

        # reorganize results Dataset:
        results = results.assign_coords({FC.VARS: out_vars}).to_dataset(dim=FC.VARS)

        if self._client is not None and self.progress_bar:
            progress(results.persist())

        # update data by calculation results:
        return results.compute()
    
    def finalize(self):
        """
        Finalizes the engine.
        """
        if self.cluster is not None:
            self.print("\n\nShutting down dask cluster")
            self._client.close()
            self._cluster.close()
        
        if self.progress_bar:
            self._pbar.unregister()
            
        dask.config.refresh()
        
        super().finalize()
        