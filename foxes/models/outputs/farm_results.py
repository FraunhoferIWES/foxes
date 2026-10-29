from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV

from .fdata import FarmOrderedFData

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.utils import Dict


class FarmResultsOutput(FarmOrderedFData):
    """
    Creates the farm results dataset
    """

    def __init__(
        self,
        *args: tuple[Any, ...],
        turbine_names: bool = True,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        *args
            Additional positional arguments for the base class
        turbine_names
            Add turbine names to results
        **kwargs
            Additional parameters for the base class
        """
        super().__init__(*args, **kwargs)
        self.turbine_names = turbine_names

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        results: Dict[str, Any] | None = None,
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
        results
            Existing chunk results, keys: output names

        Returns
        -------
        results
            The updated chunk results, keys: output names

        """
        results = super().calculate(
            algo=algo,
            mdata=mdata,
            fdata=fdata,
            tdata=tdata,
            results=results,
        )
        if FC.FARM_RESULTS not in results:
            fdata = results[FC.FDATA_FARM_ORDERED]

            i0 = mdata.states_i0(counter=True)
            i1 = i0 + mdata.n_states
            attrs: dict[str, Any] = dict(
                chunki_states=mdata.chunki_states,
                chunki_points=mdata.chunki_points,
                states_i0=i0,
            )

            vrs: list[str] = (
                self.variables if self.variables is not None else list(fdata.keys())
            )
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
                    dvars[FC.TNAME] = (
                        (FC.TURBINE,),
                        [t.name for t in algo.farm.turbines],
                    )

            results[FC.FARM_RESULTS] = dict(
                coords=coords,
                data_vars=dvars,
                attrs=attrs,
            )

        return results

    def update(
        self,
        algo: Algorithm,
        new_chunk_results: Dict[tuple[int, int], dict[str, Any]],
        all_chunk_results: Dict[tuple[int, int], dict[str, Any]],
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
        all_chunk_results: Dict[tuple[int, int], dict[str, Any]],
        output_results: Dict[str, Any],
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
        assert j0 == 0 and j1 == 1, (
            f"{self.name}: Expected exactly one target chunk, got {j0} to {j1}"
        )

        # combine results:
        coords: dict[str, np.ndarray] = {}
        dvars: dict[str, np.ndarray] = {}
        attrs: Dict[str, Any] = {}
        for i in range(i0, i1):
            fr = all_chunk_results[(i, 0)].pop(FC.FARM_RESULTS)
            for v in list(fr.keys()):
                r = fr.pop(v)

                # update coords:
                if i == i0:
                    attrs.update(r.pop("attrs"))
                    coords.update(r.pop("coords"))
                    if FC.STATE in coords:
                        coords[FC.STATE] = [coords[FC.STATE]]
                elif FC.STATE in coords:
                    coords[FC.STATE].append(r.pop(FC.STATE))

                # update data_vars:
                if i == i0:
                    for v, (dms, d) in r.pop("data_vars").items():
                        if FC.STATE in dms:
                            dvars[v] = (dms, [d])
                        else:
                            dvars[v] = (dms, d)
                else:
                    for v, (dms, d) in r.pop("data_vars").items():
                        if FC.STATE in dms:
                            assert dms == dvars[v][0], (
                                f"{self.name}: Dimensions mismatch for variable '{v}', expected {dvars[v][0]} got {dms}"
                            )
                            dvars[v][1].append(d)

        # combine states:
        if i1 == i0 + 1:
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
        output_results[FC.FARM_RESULTS] = ds
