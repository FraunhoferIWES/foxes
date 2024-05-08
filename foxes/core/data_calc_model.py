import numpy as np
import xarray as xr
from abc import abstractmethod
from dask.distributed import progress
from dask.diagnostics import ProgressBar

from .model import Model
from .data import MData, FData, TData
from foxes.utils.runners import DaskRunner
import foxes.constants as FC
import foxes.variables as FV


class DataCalcModel(Model):
    """
    Abstract base class for models with
    that run calculation on xarray Dataset
    data.

    The calculations are run via xarray's
    `apply_ufunc` function, i.e., they run in
    parallel depending on the dask settings.

    For each individual data chunk the `calculate`
    function is called.

    :group: core

    """

    @abstractmethod
    def calculate(self, algo, *data, **parameters):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of foxes.core.Data
            The input data
        parameters: dict, optional
            The calculation parameters

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray

        """
        pass

    def _wrap_calc(
        self,
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
        out_dims,
        calc_pars,
        init_vars,
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
        for li, l in enumerate(out_dims):
            for i, dims in enumerate(ldims):
                if l in dims:
                    oshape.append(ldata[i].shape[dims.index(l)])
                    break
            if len(oshape) != li + 1:
                raise ValueError(f"Model '{self.name}': Failed to find loop dimension")

        # add zero output data arrays:
        odims = {v: tuple(out_dims) for v in out_vars}
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
        self.ensure_variables(algo, *data)
        results = self.calculate(algo, *data, **calc_pars)

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
                    f"Model '{self.name}': Missing results {list(missing)}, expected shape {oshape}"
                )
        del data

        # create output:
        n_vars = len(out_vars)
        data = np.zeros(oshape + [n_vars], dtype=FC.DTYPE)
        for v in out_vars:
            data[..., out_vars.index(v)] = results[v]

        return data

    def run_calculation(
        self,
        algo,
        *data,
        out_vars,
        loop_dims,
        out_core_vars,
        initial_results=None,
        sel=None,
        isel=None,
        **calc_pars,
    ):
        """
        Starts the model calculation in parallel, via
        xarray's `apply_ufunc`.

        Typically this function is called by algorithms.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: tuple of xarray.Dataset
            The input data
        out_vars: list of str
            The calculation output variables
        loop_dims: array_like of str
            List of the loop dimensions during xarray's
            `apply_ufunc` calculations
        out_core_vars: list of str
            The core dimensions of the output data, use
            `FC.VARS` for variables dimension (required)
        initial_results: xarray.Dataset, optional
            Initial results
        sel: dict, optional
            Selection of loop_dim variable subset values
        isel: dict, optional
            Selection of loop_dim variable subset index values
        calc_pars: dict, optional
            Additional arguments for the `calculate` function

        Returns
        -------
        results: xarray.Dataset
            The calculation results

        """
        # check:
        if not self.initialized:
            raise ValueError(
                f"DataCalcModel '{self.name}': run_calculation called for uninitialized model"
            )

        # prepare:
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
        if initial_results is not None:
            ds = initial_results
            hvarsl = [v for v, d in ds.items() if len(loopd.intersection(d.dims))]
            ldata += [ds[v] for v in hvarsl]
            idims += [ds[v].dims for v in hvarsl]
            ivars += hvarsl

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
                    ldata.append(xr.DataArray(data=d.values, coords={c: d}, dims=[c]))
                    ldims.append((c,))
                    lvars.append(c)
                else:
                    edata.append(d.values)
                    edims.append((c,))
                    evars.append(c)

            dvars.append(list(ds.keys()) + list(ds.coords.keys()))

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
        if FC.TURBINE in loopd and FC.TURBINE not in ldims.values():
            dargs["output_sizes"][FC.TURBINE] = algo.n_turbines
        if FC.VARS not in out_core_vars:
            raise ValueError(
                f"Model '{self.name}': Expecting '{FC.VARS}' in out_core_vars, got {out_core_vars}"
            )

        # setup arguments for wrapper function:
        out_dims = loop_dims + list(set(out_core_vars).difference([FC.VARS]))
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
            out_dims=out_dims,
            calc_pars=calc_pars,
            init_vars=ivars,
        )

        # run parallel computation:
        iidims = [[c for c in d if c not in loopd] for d in idims]
        icdims = [[c for c in d if c not in loopd] for d in ldims]
        results = xr.apply_ufunc(
            self._wrap_calc,
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

        if DaskRunner.is_distributed() and len(ProgressBar.active):
            progress(results.persist())

        # update data by calculation results:
        return results.compute()
