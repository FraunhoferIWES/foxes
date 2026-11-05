from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any
from xarray import Dataset

from foxes.utils import Dict, write_nc
from foxes.core import Output, get_engine
from foxes.config import config
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from pathlib import Path


def _create_dataset(
    coords,
    dvars,
    attrs,
    turbine_names=None,
    wind_farm_list=None,
    cluster_list=None,
):
    """
    Create dataset from data arrays

    Parameters
    ----------
    coords
        The coordinates for the dataset
    dvars
        The data variables for the dataset
    attrs
        The attributes for the dataset
    turbine_names
        The turbine names to include in the dataset
    wind_farm_list
        The wind farm list to include in the dataset
    cluster_list
        The cluster list to include in the dataset

    Returns
    -------
    Dataset
        The created xarray Dataset

    """
    ds = Dataset(data_vars=dvars, coords=coords, attrs=attrs)
    if turbine_names is not None:
        ds[FC.TNAME] = ((FC.TURBINE,), turbine_names)
    if wind_farm_list is not None:
        if set(wind_farm_list) != set([None]):
            ds[FC.FARM] = ((FC.TURBINE,), wind_farm_list)
    if cluster_list is not None:
        if set(cluster_list) != set([None]):
            ds[FC.CLUSTER] = ((FC.TURBINE,), cluster_list)
    if FV.ORDER in ds:
        ds[FV.ORDER] = ds[FV.ORDER].astype(config.dtype_int)
    return ds


def _data_to_nc(
    fpath,
    coords,
    dvars,
    attrs,
    turbine_names=None,
    wind_farm_list=None,
    cluster_list=None,
    write_nc_pars={},
):
    """Write data arrays to a NetCDF file

    Parameters
    ----------
    fpath
        The file path to write the NetCDF file to
    coords
        The coordinates for the dataset
    dvars
        The data variables for the dataset
    attrs
        The attributes for the dataset
    turbine_names
        The turbine names to include in the dataset
    wind_farm_list
        The wind farm list to include in the dataset
    cluster_list
        The cluster list to include in the dataset
    write_nc_pars
        Additional parameters for the write_nc function
    """
    ds = _create_dataset(
        coords=coords,
        dvars=dvars,
        attrs=attrs,
        turbine_names=turbine_names,
        wind_farm_list=wind_farm_list,
        cluster_list=cluster_list,
    )
    write_nc(
        ds=ds,
        fpath=fpath,
        **write_nc_pars,
    )


class FarmResultsOutput(Output):
    """
    Creates the farm results dataset
    """

    def __init__(
        self,
        variables: list[str] | None = None,
        ambient_vars: bool = True,
        waked_vars: bool = True,
        add_turbine_names: bool = True,
        add_farm_names: bool = True,
        add_cluster_names: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        *args
            Additional positional arguments for the base class
        variables
            List of variables to include in the results.
            If None, all variables are included.
        ambient_vars
            Include ambient variables in the results
        waked_vars
            Include waked variables in the results
        add_turbine_names
            Add turbine names to results
        add_farm_names
            Add farm names to results
        add_cluster_names
            Add cluster names to results
        **kwargs
            Additional parameters for the base class
        """
        super().__init__(
            mdata_variables=[FC.STATE],
            fdata_variables=variables,
            tdata_variables=[],
            **kwargs,
        )
        self.ambient_vars = ambient_vars
        self.waked_vars = waked_vars
        self.add_turbine_names = add_turbine_names
        self.add_farm_names = add_farm_names
        self.add_cluster_names = add_cluster_names

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )
        if FC.FARM_RESULTS not in chunk_results:
            i0 = mdata.states_i0(counter=True)
            i1 = i0 + mdata.n_states
            attrs: dict[str, Any] = {}

            vrs = self.fdata_variables
            if vrs is None:
                vrs = list(fdata.keys())
            coords: dict[str, np.ndarray] = {}
            dvars: dict[str, np.ndarray] = {}
            if FC.STATE in mdata and not np.all(mdata[FC.STATE] == np.arange(i0, i1)):
                coords[FC.STATE] = mdata[FC.STATE]
            for v in vrs:
                if v in [
                    FC.STATE,
                    FC.TURBINE,
                    FC.FARM,
                    FC.CLUSTER,
                    FC.XYH,
                    FV.TXYH,
                    FC.TNAME,
                ]:
                    continue
                if not self.ambient_vars and v in FV.amb2var:
                    continue
                if not self.waked_vars and v in FV.var2amb:
                    continue
                data = fdata[v]
                dims = fdata.dims[v]
                assert dims == (FC.STATE, FC.TURBINE), (
                    f"{self.name}: Expecting dims {(FC.STATE, FC.TURBINE)} for variable '{v}', got {dims}"
                )
                if v == FV.WEIGHT and np.all(
                    np.min(data, axis=1) == np.max(data, axis=1)
                ):
                    dvars[v] = ((FC.STATE,), data[:, 0])
                else:
                    dvars[v] = ((FC.STATE, FC.TURBINE), data)

            chunk_results[FC.FARM_RESULTS] = dict(
                coords=coords,
                data_vars=dvars,
                attrs=attrs,
            )

        return chunk_results

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """
        fr = new_chunk_results.pop(FC.FARM_RESULTS)
        if FC.FARM_RESULTS not in output_results:
            output_results[FC.FARM_RESULTS] = Dict(fr, _name=FC.FARM_RESULTS)
        else:
            output_results[FC.FARM_RESULTS].update(fr)

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """
        Final update of the output results, after all chunks have been processed.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        output_results
            Output results to be updated, keys are result object names

        """
        assert n_chunks_targets == 1, (
            f"{self.name}: Expected exactly one target chunk, got {n_chunks_targets}"
        )

        # combine results:
        coords: dict[str, np.ndarray] = {}
        dvars: dict[str, np.ndarray] = {}
        attrs: dict[str, Any] = {}
        for i in range(n_chunks_states):
            fr = output_results[FC.FARM_RESULTS].pop((i, 0))

            # update coords:
            if i == 0:
                attrs.update(fr.pop("attrs"))
                coords.update(fr.pop("coords"))
                if FC.STATE in coords:
                    coords[FC.STATE] = [coords[FC.STATE]]
            elif FC.STATE in coords:
                coords[FC.STATE].append(fr.pop("coords")[FC.STATE])

            # update data_vars:
            if i == 0:
                for v, (dms, d) in fr.pop("data_vars").items():
                    if FC.STATE in dms:
                        dvars[v] = (dms, [d])
                    else:
                        dvars[v] = (dms, d)
            else:
                for v, (dms, d) in fr.pop("data_vars").items():
                    if FC.STATE in dms:
                        assert dms == dvars[v][0], (
                            f"{self.name}: Dimensions mismatch for variable '{v}', expected {dvars[v][0]} got {dms}"
                        )
                        dvars[v][1].append(d)

        # combine states:
        if n_chunks_states == 1:
            if FC.STATE in coords:
                coords[FC.STATE] = coords[FC.STATE][0]
            for v in list(dvars.keys()):
                dms = dvars[v][0]
                if FC.STATE in dms:
                    dvars[v] = (dms, dvars[v][1][0])
        else:
            if FC.STATE in coords:
                coords[FC.STATE] = np.concatenate(coords[FC.STATE], axis=0)
            for v in list(dvars.keys()):
                dms = dvars[v][0]
                if FC.STATE in dms:
                    dvars[v] = (
                        dms,
                        np.concatenate(dvars[v][1], axis=dms.index(FC.STATE)),
                    )

        # create dataset:
        tnames = algo.farm.turbine_names if self.add_turbine_names else None
        wfnames = algo.farm.wind_farm_list if self.add_farm_names else None
        cnames = algo.farm.cluster_list if self.add_cluster_names else None
        ds = _create_dataset(
            coords=coords,
            dvars=dvars,
            attrs=attrs,
            turbine_names=tnames,
            wind_farm_list=wfnames,
            cluster_list=cnames,
        )
        output_results[FC.FARM_RESULTS] = ds


class FarmResultsChunkWriter(FarmResultsOutput):
    """
    Writes one NetCDF file with farm results per chunk.
    """

    def __init__(
        self,
        base_name: str | Path,
        write_nc_pars: dict | None = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        base_name
            Base name for the NetCDF files.
        write_nc_pars
            Parameters to pass to the `write_nc` function.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        self.base_name = base_name
        self.write_nc_pars = write_nc_pars or {}
        super().__init__(**kwargs)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        fres = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=None,
        )

        coords = fres[FC.FARM_RESULTS]["coords"]
        dvars = fres[FC.FARM_RESULTS]["data_vars"]
        attrs = fres[FC.FARM_RESULTS]["attrs"]
        del fres

        attrs["base_name"] = str(self.base_name)
        attrs["chunki_states"] = mdata.chunki_states
        attrs["chunki_points"] = mdata.chunki_points
        attrs["n_chunks_states"] = mdata.n_chunks_states
        attrs["n_chunks_points"] = mdata.n_chunks_points

        if FC.STATE not in coords:
            i0 = mdata.states_i0(counter=True)
            i1 = i0 + mdata.n_states
            coords[FC.STATE] = np.arange(i0, i1)

        tnames = algo.farm.turbine_names if self.add_turbine_names else None
        wfnames = algo.farm.wind_farm_list if self.add_farm_names else None
        cnames = algo.farm.cluster_list if self.add_cluster_names else None
        ds = _create_dataset(
            coords=coords,
            dvars=dvars,
            attrs=attrs,
            turbine_names=tnames,
            wind_farm_list=wfnames,
            cluster_list=cnames,
        )

        fname = f"{self.base_name}_{mdata.chunki_states:06d}.nc"
        fpath = self.get_fpath(fname)
        write_nc(
            ds=ds,
            fpath=fpath,
            **self.write_nc_pars,
        )

        return Output.calculate(
            self,
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """
        pass

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """
        Final update of the output results, after all chunks have been processed.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        output_results
            Output results to be updated, keys are result object names

        """
        pass


class FarmResultsCountWriter(FarmResultsOutput):
    """
    Writes farm results to multiple NetCDF files based on the number of states per file.
    """

    def __init__(
        self,
        base_name: str | Path,
        n_states_per_file: int,
        write_nc_pars: dict | None = None,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        base_name
            Base name for the NetCDF files.
        n_states_per_file
            Number of states per NetCDF file.
        write_nc_pars
            Parameters to pass to the `write_nc` function.
        **kwargs
            Additional keyword arguments passed to the parent class.
        """
        self.base_name = base_name
        self.n_states_per_file = n_states_per_file
        self.write_nc_pars = write_nc_pars or {}
        super().__init__(**kwargs)

        self.FARM_RESULTS = self.var(FC.FARM_RESULTS)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )

        coords = chunk_results[FC.FARM_RESULTS]["coords"]
        if FC.STATE not in coords:
            i0 = mdata.states_i0(counter=True)
            i1 = i0 + mdata.n_states
            coords[FC.STATE] = np.arange(i0, i1)

        attrs = chunk_results[FC.FARM_RESULTS]["attrs"]
        attrs["base_name"] = str(self.base_name)
        if self.n_states_per_file > 0:
            attrs["n_states_per_file"] = self.n_states_per_file

        return chunk_results

    def _init_output_results(self, first_chunk_results):
        """Helper function to initialize output results for farm results"""
        output_results = Dict({}, _name=self.FARM_RESULTS)
        output_results["key"] = (0, 0)
        output_results["index"] = 0
        output_results["futures"] = []
        return output_results

    def _write_criteria(self, results) -> bool:
        """Helper function for determining the write criteria"""
        return results["attrs"]["n_states"] >= self.n_states_per_file

    def _reduce_and_select(
        self,
        key,
        n_chunks_states,
        results,
        coords,
        dvars,
        attrs,
        hstates=None,
    ):
        """Helper function for reducing and selecting farm results"""
        if key[0] < n_chunks_states - 1:
            if hstates is None:
                hstates = min(self.n_states_per_file, len(coords[FC.STATE]))

            # reduce accumulated results:
            results["coords"][FC.STATE] = [coords[FC.STATE][hstates:]]
            for v in list(dvars.keys()):
                dms, d = dvars[v]
                if FC.STATE in dms:
                    results["data_vars"][v] = (dms, [d[hstates:, ...]])

            # select the first n_states_per_file states from the accumulated results:
            coords = {c: d if c != FC.STATE else d[:hstates] for c, d in coords.items()}
            dvars = {
                v: (dms, d if FC.STATE not in dms else d[:hstates])
                for v, (dms, d) in dvars.items()
            }

        index = results["index"]
        hstates = len(coords[FC.STATE])
        attrs = results["attrs"].copy()
        attrs["n_states"] = hstates
        attrs["index"] = index
        results["attrs"]["n_states"] -= hstates

        return coords, dvars, attrs

    def _get_fpath(self, attrs):
        """Helper function to get the file path for farm results"""
        index = attrs["index"]
        fname = f"{self.base_name}_{index:06d}.nc"
        return self.get_fpath(fname)

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """
        Update output results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        new_chunk_results
            New chunk results, keys are result object names.
            Values are dicts with keys (states_chunk_index, target_chunk_index)
        output_results
            Output results to be updated, keys are result object names

        """
        engine = get_engine()

        fr = new_chunk_results[FC.FARM_RESULTS]
        if self.FARM_RESULTS not in output_results:
            output_results[self.FARM_RESULTS] = self._init_output_results(fr[(0, 0)])

        results = output_results[self.FARM_RESULTS]
        for k in list(fr.keys()):
            key = results["key"]
            assert k == key, f"{self.name}: unexpected key {k}, expected {key}"
            chunk_results = fr.pop(k)

            # add initial results:
            if key == (0, 0):
                results["coords"] = chunk_results["coords"].copy()
                results["data_vars"] = chunk_results["data_vars"].copy()
                results["attrs"] = chunk_results["attrs"].copy()

                results["attrs"]["n_states"] = len(chunk_results["coords"][FC.STATE])
                results["coords"][FC.STATE] = [chunk_results["coords"][FC.STATE]]
                for v in list(chunk_results["data_vars"].keys()):
                    dms, d = chunk_results["data_vars"][v]
                    if FC.STATE in dms:
                        assert dms.index(FC.STATE) == 0, (
                            f"{self.name}: expected {FC.STATE} to be the first dimension in {v}, got {dms.index(FC.STATE)}"
                        )
                        results["data_vars"][v] = (dms, [d])

            # add subsequent results:
            else:
                results["attrs"]["n_states"] += len(chunk_results["coords"][FC.STATE])
                results["coords"][FC.STATE].append(chunk_results["coords"][FC.STATE])
                for v in list(chunk_results["data_vars"].keys()):
                    dms, d = chunk_results["data_vars"][v]
                    if FC.STATE in dms:
                        assert dms == results["data_vars"][v][0], (
                            f"{self.name}: expected dimensions {results['data_vars'][v][0]} for {v}, got {dms}"
                        )
                        results["data_vars"][v][1].append(d)

            while self._write_criteria(results) or (
                key[0] == n_chunks_states - 1 and results["attrs"]["n_states"] > 0
            ):
                # concat results:
                coords = {
                    c: d if c != FC.STATE else np.concatenate(results["coords"][c])
                    for c, d in results["coords"].items()
                }
                dvars = {
                    v: (dms, d)
                    if FC.STATE not in dms
                    else (dms, np.concatenate(results["data_vars"][v][1], axis=0))
                    for v, (dms, d) in results["data_vars"].items()
                }
                attrs = results["attrs"].copy()

                coords, dvars, attrs = self._reduce_and_select(
                    key, n_chunks_states, results, coords, dvars, attrs
                )

                # write dataset:
                fpath = self._get_fpath(attrs)
                tnames = algo.farm.turbine_names if self.add_turbine_names else None
                wfnames = algo.farm.wind_farm_list if self.add_farm_names else None
                cnames = algo.farm.cluster_list if self.add_cluster_names else None
                results["futures"].append(
                    engine.submit(
                        _data_to_nc,
                        fpath,
                        coords,
                        dvars,
                        attrs,
                        tnames,
                        wfnames,
                        cnames,
                    )
                )

                results["index"] += 1
            results["key"] = (key[0] + 1, 0)

            if results["key"][0] == n_chunks_states:
                results.pop("coords")
                results.pop("data_vars")
                results.pop("attrs")
                results.pop("key")
                results.pop("index")

        return output_results

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """
        Final update of the output results, after all chunks have been processed.

        Parameters
        ----------
        algo
            The algorithm
        n_chunks_states
            The number of state chunks
        n_chunks_targets
            The number of target chunks
        output_results
            Output results to be updated, keys are result object names

        """
        engine = get_engine()
        fr = output_results.pop(self.FARM_RESULTS)
        futures = fr.pop("futures")
        for f in futures:
            engine.await_result(f)


class FarmResultsPeriodWriter(FarmResultsCountWriter):
    """
    Writes farm results per time period to multiple NetCDF files.
    """

    def __init__(
        self,
        *args,
        period: str["day", "week", "month", "year"],
        **kwargs,
    ):
        super().__init__(*args, n_states_per_file=-1, **kwargs)
        self.period = period

        assert self.period in ["day", "week", "month", "year"], (
            f"{self.name}: Invalid period {self.period}, must be one of ['day', 'week', 'month', 'year']"
        )

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        The chunk based output calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data
        model_results
            The results from the model calculation for this chunk
        chunk_results
            Existing chunk results, keys are result object names

        Returns
        -------
        chunk_results
            The updated chunk results, keys are result object names

        """
        chunk_results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            model_results=model_results,
            chunk_results=chunk_results,
        )

        attrs = chunk_results[FC.FARM_RESULTS]["attrs"]
        attrs["period"] = self.period

        return chunk_results

    def _init_output_results(self, first_chunk_results):
        """Helper function to initialize output results for farm results"""
        output_results = super()._init_output_results(first_chunk_results)

        time0 = first_chunk_results["coords"][FC.STATE][0]
        assert isinstance(time0, np.datetime64), (
            f"{self.name}: State coordinates must be of type np.datetime64, got {type(time0)}"
        )

        return output_results

    def _write_criteria(self, results) -> bool:
        """Helper function for determining the write criteria"""
        if results["attrs"]["n_states"] <= 0:
            return False
        first_time = results["coords"][FC.STATE][0][0].astype("datetime64[D]")
        latest_time = results["coords"][FC.STATE][-1][-1].astype("datetime64[D]")
        if self.period == "day":
            next_time = first_time.astype("datetime64[D]") + np.timedelta64(1, "D")
        if self.period == "week":
            next_time = first_time.astype("datetime64[D]") + np.timedelta64(1, "W")
        if self.period == "month":
            next_time = first_time.astype("datetime64[M]") + np.timedelta64(1, "M")
        if self.period == "year":
            next_time = first_time.astype("datetime64[Y]") + np.timedelta64(1, "Y")
        results["time"] = next_time
        return latest_time >= next_time

    def _reduce_and_select(
        self,
        key,
        n_chunks_states,
        results,
        coords,
        dvars,
        attrs,
        hstates=None,
    ):
        """Helper function for reducing and selecting farm results"""
        hstates = np.searchsorted(coords[FC.STATE], results["time"], side="left")

        coords, dvars, attrs = super()._reduce_and_select(
            key, n_chunks_states, results, coords, dvars, attrs, hstates=hstates
        )

        time0 = coords[FC.STATE][0].astype("datetime64[D]").astype(object)
        attrs["year"] = time0.year
        attrs["month"] = time0.month
        attrs["week"] = time0.isocalendar().week
        attrs["day"] = time0.day

        return coords, dvars, attrs

    def _get_fpath(self, attrs):
        """Helper function to get the file path for farm results"""
        if self.period == "day":
            fname = f"{self.base_name}_{attrs['year']}-{attrs['month']:02d}-{attrs['day']:02d}.nc"
        elif self.period == "week":
            fname = f"{self.base_name}_{attrs['year']}-W{attrs['week']:02d}.nc"
        elif self.period == "month":
            fname = f"{self.base_name}_{attrs['year']}-{attrs['month']:02d}.nc"
        elif self.period == "year":
            fname = f"{self.base_name}_{attrs['year']}.nc"
        return self.get_fpath(fname)
