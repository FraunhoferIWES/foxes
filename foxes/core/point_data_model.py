import xarray as xr
import numpy as np
from abc import abstractmethod
from dask.distributed import progress

from foxes.core.model import Model
from foxes.core.data import Data
import foxes.variables as FV
import foxes.constants as FC

class PointDataModel(Model):

    @abstractmethod
    def output_point_vars(self, algo):
        return []
    
    @abstractmethod
    def calculate(self, algo, mdata, fdata, pdata):
        pass

    def _wrap_calc(
        self,
        *data,
        algo,
        idims,
        edata,
        edims,
        ovars,
        mkeys,
        fkeys,
        pkeys,
        calc_pars
    ):
        # extract models data:
        mdata = {v: data[i] for i, v in enumerate(idims.keys()) if v in mkeys}
        mdata.update({v: d for v, d in edata.items() if v in mkeys})
        mdims = {v: d for v, d in idims.items() if v in mkeys}
        mdims.update({v: d for v, d in edims.items() if v in mkeys})
        mdata = Data(mdata, mdims, loop_dims=[FV.STATE, FV.POINT])
        del mdims

        # extract farm data:
        fdata = {v: data[i] for i, v in enumerate(idims.keys()) if v in fkeys}
        fdata.update({v: d for v, d in edata.items() if v in fkeys})
        fdims = {v: d for v, d in idims.items() if v in fkeys}
        fdims.update({v: d for v, d in edims.items() if v in fkeys})
        fdata = Data(fdata, fdims, loop_dims=[FV.STATE, FV.POINT])
        del fdims

        # extract point data:
        pdata = {v: data[i] for i, v in enumerate(idims.keys()) if v in pkeys}
        pdata.update({v: d for v, d in edata.items() if v in pkeys})
        pdims = {v: d for v, d in idims.items() if v in pkeys}
        pdims.update({v: d for v, d in edims.items() if v in pkeys})
        n_states = mdata.n_states
        n_points = len(pdata[FV.POINT])
        pdata.update({v: np.full((n_states, n_points), np.nan, dtype=FC.DTYPE) for v in ovars})
        pdims.update({v: (FV.STATE, FV.POINT) for v in ovars})
        pdata = Data(pdata, pdims, loop_dims=[FV.STATE, FV.POINT])
        del pdims, data, edata, idims, edims

        # run model calculation:
        results = self.calculate(algo, mdata, fdata, pdata, **calc_pars)
        del mdata, fdata, pdata
        
        # create output:
        n_vars = len(ovars)
        data   = np.zeros((n_states, n_points, n_vars), dtype=FC.DTYPE)
        for v in ovars:
            data[:, :, ovars.index(v)] = results[v]
        
        return data

    def run_calculation(self, algo, models_data, farm_data, point_data, ovars, **parameters):

        if not self.initialized:
            raise ValueError(f"PointDataModel '{self.name}': run_calc called before initialization")

        # collect models data:
        idata = {v: d for v, d in models_data.items() if FV.STATE in d.dims or FV.POINT in d.dims}
        edata = {v: d.to_numpy() for v, d in models_data.items() if v not in idata}
        mkeys = list(models_data.keys())

        # collect farm data:
        idata.update({v: d for v, d in farm_data.items() if FV.STATE in d.dims or FV.POINT in d.dims})
        edata.update({v: d.to_numpy() for v, d in farm_data.items() if v not in idata})
        fkeys = list(farm_data.keys())
        
        # collect point data:
        idata.update({v: d for v, d in point_data.items() if FV.STATE in d.dims or FV.POINT in d.dims})
        edata.update({v: d.to_numpy() for v, d in point_data.items() if v not in idata})
        pkeys = list(point_data.keys())

        # add states:
        states = None
        for d in farm_data.values():
            for ci, c in enumerate(d.dims):
                if c == FV.STATE:
                    crds   = {c: farm_data.coords[c]} if c in farm_data.coords else None
                    states = xr.DataArray(data=farm_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        states = states.chunk(d.chunks[ci])
                    break
            if states is not None:
                idata[FV.STATE] = states
                if not FV.STATE in mkeys:
                    mkeys.append(FV.STATE)
                break
        if states is None:
            raise ValueError(f"FarmDataModel '{self.name}': Missing dimension '{FV.STATE}' in farm data coordinates.")
        
        # add points:
        points = None
        for d in point_data.values():
            for ci, c in enumerate(d.dims):
                if c == FV.POINT:
                    crds   = {c: point_data.coords[c]} if c in point_data.coords else None
                    points = xr.DataArray(data=point_data[c].to_numpy(), coords=crds, dims=[c])
                    if d.chunks is not None:
                        points = points.chunk(d.chunks[ci])
                    break
            if points is not None:
                idata[FV.POINT] = points
                if not FV.POINT in pkeys:
                    pkeys.append(FV.POINT)
                break
        if points is None:
            raise ValueError(f"FarmDataModel '{self.name}': Missing dimension '{FV.POINT}' in point data coordinates.")

        # collect dims:
        idims  = {v: d.dims for v, d in idata.items()}
        icdims = [[c for c in d if c != FV.STATE and c != FV.POINT] for d in idims.values()] 
        ocdims = [[FV.SP_VARS]]
        otypes = [FC.DTYPE]
        edims  = {}
        for v in edata.keys():
            for s in (models_data, farm_data, point_data):
                if v in s:
                    edims[v] = s[v].dims
                    break

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
            mkeys=mkeys,
            fkeys=fkeys,
            pkeys=pkeys,
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