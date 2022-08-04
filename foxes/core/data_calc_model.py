import numpy as np
import xarray as xr
from abc import abstractmethod
from dask.distributed import progress

from .model import Model
from .data import Data
import foxes.variables as FV
import foxes.constants as FC


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

    """

    @abstractmethod
    def calculate(self, algo, *data, **parameters):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        *data : foxes.core.Data
            The input data
        **parameters : dict, optional
            The calculation parameters

        Returns
        -------
        results : dict
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
    ):
        """
        Wrapper that mitigates between apply_ufunc and `calculate`.
        """

        # reconstruct original data:
        data = []
        for hvars in dvars:

            v2l = {v: lvars.index(v) for v in hvars if v in lvars}
            v2e = {v: evars.index(v) for v in hvars if v in evars}

            hdata = {v: ldata[v2l[v]] if v in v2l else edata[v2e[v]] for v in hvars}
            hdims = {v: ldims[v2l[v]] if v in v2l else edims[v2e[v]] for v in hvars}

            data.append(Data(hdata, hdims, loop_dims))

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
        odims = {v: out_dims for v in out_vars}
        odata = {
            v: np.full(oshape, np.nan, dtype=FC.DTYPE)
            for v in out_vars
            if v not in data[-1]
        }
        if len(data) == 1:
            data.append(Data(odata, odims, loop_dims))
        else:
            odata.update(data[-1])
            odims.update(data[-1].dims)
            data[-1] = Data(odata, odims, loop_dims)
        del odims, odata

        # run model calculation:
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
        self, algo, *data, out_vars, loop_dims, out_core_vars, **calc_pars
    ):
        """
        Starts the model calculation in parallel, via
        xarray's `apply_ufunc`.

        Typically this function is called by algorithms.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        *data : tuple of xarray.Dataset
            The input data
        out_vars: list of str
            The calculation output variables
        loop_dims : array_like of str
            List of the loop dimensions during xarray's
            `apply_ufunc` calculations
        out_core_vars : list of str
            The core dimensions of the output data, use
            `FV.VARS` for variables dimension (required)
        **calc_pars : dict, optional
            Additional arguments for the `calculate` function

        Returns
        -------
        results : xarray.Dataset
            The calculation results

        """

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

        # setup dask options:
        dargs = dict(output_sizes={FV.VARS: len(out_vars)})
        if FV.TURBINE in loopd and FV.TURBINE not in ldims.values():
            dargs["output_sizes"][FV.TURBINE] = algo.n_turbines
        if FV.VARS not in out_core_vars:
            raise ValueError(
                f"Model '{self.name}': Expecting '{FV.VARS}' in out_core_vars, got {out_core_vars}"
            )

        # setup arguments for wrapper function:
        out_dims = loop_dims + list(set(out_core_vars).difference([FV.VARS]))
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
        )

        # run parallel computation:
        icdims = [[c for c in d if c not in loopd] for d in ldims]
        results = xr.apply_ufunc(
            self._wrap_calc,
            *ldata,
            input_core_dims=icdims,
            output_core_dims=[out_core_vars],
            output_dtypes=[FC.DTYPE],
            dask="parallelized",
            dask_gufunc_kwargs=dargs,
            kwargs=wargs,
        )

        # reorganize results Dataset:
        results = (
            results.assign_coords({FV.VARS: out_vars}).to_dataset(dim=FV.VARS).persist()
        )

        # try to show progress bar:
        try:
            progress(results)
        except ValueError:
            pass

        # update data by calculation results:
        return results.compute()
