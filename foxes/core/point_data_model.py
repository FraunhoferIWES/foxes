import xarray as xr
import numpy as np
from abc import abstractmethod
from dask.distributed import progress

from foxes.core.model import Model
from foxes.core.farm_data import FarmData
from foxes.core.point_data import PointData
import foxes.variables as FV
import foxes.constants as FC

class PointDataModel(Model):

    def input_point_data(self, algo):
        return {"coords": {}, "data_vars": {}}

    @abstractmethod
    def output_point_vars(self, algo):
        return []

    def initialize(self, algo, farm_data, point_data):
        super().initialize()
    
    @abstractmethod
    def calculate(self, algo, fdata, pdata):
        pass

    def finalize(self, algo, farm_data, point_data):
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
        pdata = {v: data[i] for i, v in enumerate(idims.keys()) if FV.POINT in idims[v]}
        fdata = {v: data[i] for i, v in enumerate(idims.keys()) if v not in pdata}
        fdata.update(edata)
        n_states = len(fdata[FV.STATE])
        n_points = len(pdata[FV.POINT])
        del data, edata

        # collect dimensions info:
        dims = {v: (FV.STATE, FV.POINT) for v in ovars}
        dims.update(idims)
        dims.update(edims)
        del idims, edims

        # run model calculation:
        fdata = FarmData(fdata, {v: d for v, d in dims.items() if v in fdata}, algo.n_turbines)
        pdata = PointData(pdata, {v: d for v, d in dims.items() if v in pdata})
        hres  = self.calculate(algo=algo, fdata=fdata, pdata=pdata, **calc_pars)
        ores  = {v: d for v, d in pdata.items() if v not in hres}
        del fdata, pdata, dims
        
        # create output:
        n_vars = len(ovars)
        data   = np.zeros((n_states, n_points, n_vars), dtype=FC.DTYPE)
        for v in ovars:
            data[:, :, ovars.index(v)] = hres[v] if v in hres else ores[v]
        
        return data

    def run_calculation(self, algo, farm_data, point_data, ovars, **parameters):

        if not self.initialized:
            raise ValueError(f"Model '{self.name}': run_calc called before initialization")

        # collect farm data:
        idata = {v: d for v, d in farm_data.items() if FV.STATE in d.dims or FV.POINT in d.dims}
        edata = {v: d.to_numpy() for v, d in farm_data.items() if v not in idata}

        # collect point data:
        idata.update({v: d for v, d in point_data.items() if FV.STATE in d.dims or FV.POINT in d.dims})
        edata.update({v: d.to_numpy() for v, d in point_data.items() if v not in idata})
        otypes = [FC.DTYPE]

        # extract states data:
        states = None
        for v, d in farm_data.items():
            for ci, c in enumerate(d.dims):
                if c == FV.STATE:
                    crds   = {c: farm_data.coords[c]} if c in farm_data.coords else None
                    states = xr.DataArray(data=farm_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        states = states.chunk(d.chunks[ci])
                    break
            if states is not None:
                idata[FV.STATE] = states
                break

        # extract points data:
        points = None
        for v, d in point_data.items():
            for ci, c in enumerate(d.dims):
                if c == FV.POINT:
                    crds   = {c: point_data.coords[c]} if c in point_data.coords else None
                    points = xr.DataArray(data=point_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        points = points.chunk(d.chunks[ci])
                    break
            if points is not None:
                idata[FV.POINT] = points
                break

        # collect dims:
        idims  = {v: d.dims for v, d in idata.items()}
        edims  = {v: farm_data[v].dims if v in farm_data else point_data[v].dims for v in edata.keys()}
        icdims = [[c for c in d if c != FV.STATE and c != FV.POINT] for d in idims.values()] 
        ocdims = [[FV.SP_VARS]]

        # setup dask options:
        dargs = dict(
            output_sizes = {FV.SP_VARS: len(ovars)}
        )

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
        
        results = results.assign_coords({FV.SP_VARS: ovars}).to_dataset(dim=FV.SP_VARS).persist()

        try:
            progress(results)
        except ValueError:
            pass

        # update data by calculation results:
        return results.compute()
