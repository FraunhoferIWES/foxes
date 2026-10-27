from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import Output
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
        turbine_names: bool = True,
        ambient_vars: bool = True,
        waked_vars: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        variables
            The farm results variables, or all by default
        turbine_names
            Add turbine names to results
        ambient_vars
            Include ambient variables
        waked_vars
            Include waked variables
        **kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.variables = variables
        self.turbine_names = turbine_names
        self.ambient_vars = ambient_vars
        self.waked_vars = waked_vars

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, Any]:
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

        Returns
        -------
        results
            The resulting data, keys: output names

        """
        i0 = mdata.states_i0(counter=True)
        i1 = i0 + mdata.n_states
        attrs = dict[str: Any] = dict(
            chunki_states=mdata.chunki_states,
            chunki_points=mdata.chunki_points,
            states_i0=i0,
        )

        vrs: list[str] = self.variables if self.variables is not None else list(fdata.keys())
        coords: dict[str, np.ndarray] = {}
        dvars: dict[str, np.ndarray] = {}
        if FC.STATE in mdata and mdata[FC.STATE] != np.arange(i0, i1):
            coords[FC.STATE] = mdata[FC.STATE]
        for v in vrs:
            if not self.ambient_vars and v in FV.amb2var:
                continue
            if not self.waked_vars and v in FV.var2amb:
                continue
            data = fdata[v]
            dims = fdata.dims[v]
            assert dims == (FC.STATE, FC.TURBINE), (
                f"{self.name}: Expecting dims {(FC.STATE, FC.TURBINE)} for variable '{v}', got {dims}"
            )
            if v == FV.WEIGHT and np.min(data, axis=1) == np.max(data, axis=1):
                dvars[v] = ((FC.STATE,), data[:, 0])
            else:
                dvars[v] = ((FC.STATE, FC.TURBINE), data)

        if attrs["chunki_states"] == 0 and attrs["chunki_points"] == 0:
            if self.turbine_names:
                dvars[FC.TNAME] = ((FC.TURBINE,), [t.name for t in algo.farm.turbines])

        return {
            FC.FARM_RESULTS: dict(
                coords=coords,
                data_vars=dvars,
                attrs=attrs,
            )
        }

    def update(
        self, 
        algo: Algorithm,
        new_chunk_results: dict[tuple[int, int], dict[str, Any]], 
        all_chunk_results: dict[tuple[int, int], dict[str, Any]], 
    ) -> None:
        """
        Update all chunk results with latest new chunk results.

        Parameters
        ----------
        algo
            The algorithm
        new_chunk_results
            New chunk results, keys: (states_chunk_index, target_chunk_index),
            values: dict with output names as keys
        all_chunk_results
            Existing chunk results to be updated, 
            keys: (states_chunk_index, target_chunk_index),
            values: dict with output names as keys

        """
        for k, nr in new_chunk_results.items():
            if FC.FARM_RESULTS in nr:
                fr = nr.pop(FC.FARM_RESULTS)
                if FC.FARM_RESULTS in all_chunk_results[k]:
                    for s in list(fr.keys()):
                        all_chunk_results[k][FC.FARM_RESULTS][s].update(fr[s])
                else:
                    all_chunk_results[k][FC.FARM_RESULTS] = fr

    def aggregate(
        self, 
        algo: Algorithm, 
        all_chunk_results: dict[tuple[int, int], dict[str, Any]], 
        output_results: dict[str, Any],
    ) -> None:
        """
        Final aggregation of the collected results.

        Parameters
        ----------
        algo
            The algorithm
        all_chunk_results
            All chunk results, keys: (states_chunk_index, target_chunk_index),
            values: dict with output names as keys
        output_results
            Existing aggregated output results to be updated, keys: Results names

        """
        # find states and target chunk indices:
        (i0, i1), (j0, j1) = self.find_chunk_ranges(all_chunk_results)

        # combine results:
        coords: dict[str, np.ndarray] = {}
        dvars: dict[str, np.ndarray] = {}
        for i in range(i0, i1):
            for j in range(j0, j1):
                fr = all_chunk_results[(i, j)].pop(FC.FARM_RESULTS)
                for v in list(fr.keys()):
                    r = fr.pop(v)
                    TODO
                    


