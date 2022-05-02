import xarray as xr
import numpy as np
from abc import abstractmethod
from dask.distributed import progress

import foxes.constants as FC
import foxes.variables as FV
from foxes.core.data import Data
from foxes.core.model import Model

class FarmDataModel(Model):

    @abstractmethod
    def output_farm_vars(self, algo):
        return []

    @abstractmethod
    def calculate(self, algo, mdata, fdata):
        pass

    def _wrap_calc(
        self,
        *data,
        algo,
        idims,
        edata,
        edims,
        ovars,
        calc_pars
    ):
        # extract model data:
        mdata = {v: data[i] for i, v in enumerate(idims.keys())}
        mdata.update(edata)
        n_states = len(mdata[FV.STATE])
        idims.update(edims)
        mdata = Data(mdata, idims, loop_dims=[FV.STATE])
        del edata, edims

        # create zero output data:
        dims  = {v: (FV.STATE, FV.TURBINE) for v in ovars}
        fdata = {v: np.full((n_states, algo.n_turbines), np.nan, dtype=FC.DTYPE) \
                    for v in ovars if v not in idims.keys()}
        for v in set(ovars).intersection(set(idims.keys())):
            if idims[v] == (FV.STATE, FV.TURBINE):
                fdata[v] = data[list(idims.keys()).index(v)]
            else:
                raise ValueError(f"Wrong dimension for output variable '{v}': Expected {(FV.STATE, FV.TURBINE)}, got {idims[v]}")
        fdata = Data(fdata, dims, loop_dims=[FV.STATE])
        del dims, idims, data

        # run model calculation:
        self.calculate(algo, mdata, fdata, **calc_pars)
        del mdata
        
        # create output:
        n_vars = len(ovars)
        data   = np.zeros((n_states, algo.n_turbines, n_vars), dtype=FC.DTYPE)
        for v in ovars:
            data[:, :, ovars.index(v)] = fdata[v]
        
        return data
            
    def run_calculation(self, algo, models_data, **parameters):

        if not self.initialized:
            raise ValueError(f"FarmDataModel '{self.name}': run_calc called before initialization")

        # collect models data:
        idata  = {v: d for v, d in models_data.items() if FV.STATE in d.dims}
        edata  = {v: d.to_numpy() for v, d in models_data.items() if v not in idata}
        otypes = [FC.DTYPE]
        ovars  = algo.farm_vars 
        if not FV.WEIGHT in ovars:
            ovars.append(FV.WEIGHT)

        # extract states data:
        states = None
        for d in models_data.values():
            for ci, c in enumerate(d.dims):
                if c == FV.STATE:
                    crds   = {FV.STATE: models_data.coords[c]} if c in models_data.coords else None
                    states = xr.DataArray(data=models_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        states = states.chunk(d.chunks[ci])
                    break
            if states is not None:
                idata[FV.STATE] = states
                break
        if states is None:
            raise ValueError(f"FarmDataModel '{self.name}': Missing dimension '{FV.STATE}' in models data coordinates.")

        # collect dims:
        idims  = {v: d.dims for v, d in idata.items()}
        edims  = {v: models_data[v].dims for v in edata.keys()}
        icdims = [[c for c in d if c != FV.STATE] for d in idims.values()] 
        ocdims = [[FV.TURBINE, FV.ST_VARS]]

        # setup dask options:
        dargs = dict(
            output_sizes = {FV.ST_VARS: len(ovars)}
        )
        if FV.TURBINE not in ovars:
            dargs["output_sizes"][FV.TURBINE] = algo.n_turbines

        # setup arguments for wrapper function:
        wargs = dict(
            algo=algo,
            idims=idims,
            edata=edata,
            edims=edims,
            ovars=ovars,
            calc_pars=parameters
        )

        # run parallel computation:
        results = xr.apply_ufunc(
                    self._wrap_calc, 
                    *idata.values(), 
                    input_core_dims=icdims, 
                    output_core_dims=ocdims, 
                    output_dtypes=otypes,
                    dask="parallelized",
                    dask_gufunc_kwargs=dargs,
                    kwargs=wargs
                )
        
        results = results.assign_coords({FV.ST_VARS: ovars}).to_dataset(dim=FV.ST_VARS).persist()

        try:
            progress(results)
        except ValueError:
            pass

        # update data by calculation results:
        return results.compute()
