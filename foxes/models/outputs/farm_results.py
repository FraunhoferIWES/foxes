from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any
from xarray import Dataset

from foxes.utils import Dict
import foxes.constants as FC
import foxes.variables as FV

from .fdata import FarmOrderedFData

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


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
        model_results: Any,
        chunk_results: Dict | None = None,
    ) -> Dict:
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
            Existing chunk results

        Returns
        -------
        chunk_results
            The updated chunk results

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
            fdata = chunk_results[FC.FDATA_FARM_ORDERED]

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
            if FC.STATE in mdata and not np.all(mdata[FC.STATE] == np.arange(i0, i1)):
                coords[FC.STATE] = mdata[FC.STATE]
            for v in vrs:
                if v in [FC.STATE, FC.TURBINE, FC.XYH, FV.TXYH]:
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

            if attrs["chunki_states"] == 0 and attrs["chunki_points"] == 0:
                if self.turbine_names:
                    dvars[FC.TNAME] = (
                        (FC.TURBINE,),
                        [t.name for t in algo.farm.turbines],
                    )

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
        new_chunk_results: Dict[tuple[int, int], Dict],
        output_results: Dict,
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
            New chunk results, keys: (states_chunk_index, target_chunk_index),
            values: dict with chunk results
        output_results
            Existing output results to be updated

        """
        for k, nr in new_chunk_results.items():
            if FC.FARM_RESULTS in nr:
                fr = nr.pop(FC.FARM_RESULTS)
                if FC.FARM_RESULTS not in output_results:
                    output_results[FC.FARM_RESULTS] = Dict(_name=FC.FARM_RESULTS)
                    output_results[FC.FARM_RESULTS][k] = fr
                elif k not in output_results[FC.FARM_RESULTS]:
                    output_results[FC.FARM_RESULTS][k] = fr
                else:
                    for s in list(fr.keys()):
                        output_results[FC.FARM_RESULTS][k][s].update(fr[s])

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict,
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
            Existing output results to be updated

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
        output_results[FC.FARM_RESULTS] = ds
