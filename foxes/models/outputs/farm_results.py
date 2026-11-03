from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any
from xarray import Dataset

from foxes.utils import Dict
from foxes.core import Output
from foxes.config import config
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


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
        ds = Dataset(data_vars=dvars, coords=coords, attrs=attrs)
        if self.add_turbine_names:
            ds[FC.TNAME] = ((FC.TURBINE,), algo.farm.turbine_names)
        if self.add_farm_names:
            fnames = algo.farm.wind_farm_list
            if set(fnames) != set([None]):
                ds[FC.FARM] = ((FC.TURBINE,), fnames)
        if self.add_cluster_names:
            cnames = algo.farm.cluster_list
            if set(cnames) != set([None]):
                ds[FC.CLUSTER] = ((FC.TURBINE,), cnames)
        if FV.ORDER in ds:
            ds[FV.ORDER] = ds[FV.ORDER].astype(config.dtype_int)

        output_results[FC.FARM_RESULTS] = ds
