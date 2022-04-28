import xarray as xr
import numpy as np
from abc import abstractmethod
from dask.distributed import progress

import foxes.constants as FC
import foxes.variables as FV
from foxes.core.farm_data import FarmData
from foxes.core.model import Model

class FarmDataModel(Model):
    
    def input_farm_data(self, algo):
        return {"coords": {}, "data_vars": {}}

    @abstractmethod
    def output_farm_vars(self, algo):
        return []

    def initialize(self, algo, farm_data):
        super().initialize()

    @abstractmethod
    def calculate(self, algo, fdata):
        pass

    def finalize(self, algo, farm_data):
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
        # extract data into dicts, for better accessability:
        fdata = {v: data[i] for i, v in enumerate(idims.keys())}
        fdata.update(edata)
        n_states = len(fdata[FV.STATE])
        del data, edata

        # collect dimensions info:
        dims = {v: (FV.STATE, FV.TURBINE) for v in ovars}
        dims.update(idims)
        dims.update(edims)
        del idims, edims

        # run model calculation:
        fdata = FarmData(fdata, dims, algo.n_turbines)
        hres  = self.calculate(algo, fdata, **calc_pars)
        ores  = {v: d for v, d in fdata.items() if v not in hres}
        del fdata, dims
        
        # create output:
        n_vars = len(ovars)
        data   = np.zeros((n_states, algo.n_turbines, n_vars), dtype=FC.DTYPE)
        for v in ovars:
            data[:, :, ovars.index(v)] = hres[v] if v in hres else ores[v]
        
        return data
            
    def run_calculation(self, algo, farm_data, **parameters):

        if not self.initialized:
            raise ValueError(f"FarmDataModel '{self.name}': run_calc called before initialization")

        # collect data:
        idata  = {v: d for v, d in farm_data.items() if FV.STATE in d.dims}
        edata  = {v: d.to_numpy() for v, d in farm_data.items() if v not in idata}
        otypes = [FC.DTYPE]
        ovars  = algo.farm_vars 
        if not FV.WEIGHT in ovars:
            ovars.append(FV.WEIGHT)

        # extract states data:
        states = None
        for v, d in farm_data.items():
            for ci, c in enumerate(d.dims):
                if c == FV.STATE:
                    crds   = {FV.STATE: farm_data.coords[c]} if c in farm_data.coords else None
                    states = xr.DataArray(data=farm_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        states = states.chunk(d.chunks[ci])
                    break
            if states is not None:
                idata[FV.STATE] = states
                break

        # collect dims:
        idims  = {v: d.dims for v, d in idata.items()}
        edims  = {v: farm_data[v].dims for v in edata.keys()}
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
