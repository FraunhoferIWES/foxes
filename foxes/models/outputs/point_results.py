"""Collect point model results through the engine output lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

from foxes.core import Output
from foxes.utils import Dict
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import Algorithm, MData, FData, TData


class PointResultsOutput(Output):
    """Unify state/target chunks without expanding static input coordinates."""

    def __init__(self, variables: list[str], **kwargs: Any) -> None:
        """Select point fields to collect.

        Parameters
        ----------
        variables
            Point model result names, including weights when requested.
        kwargs
            Common output configuration.
        """
        # Point collectors read prepared target data and model return values.
        super().__init__(
            mdata_variables=[FC.STATE],
            fdata_variables=[],
            tdata_variables=variables,
            **kwargs,
        )
        self.variables = list(variables)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect selected NumPy point fields for one state/target chunk.

        Parameters
        ----------
        algo
            Calculation algorithm.
        mdata
            Chunk metadata and state coordinates.
        fdata
            Farm data, unused by this collector.
        tdata
            Target data; required for point results.
        model_results
            Point model return mapping.
        chunk_results
            Shared output mapping or ``None``.

        Returns
        -------
        chunk_results
            Shared mapping with point arrays and explicit chunk coordinates.

        Raises
        ------
        ValueError
            If no target data is supplied.
        """
        # Preserve state and global target labels independently of chunk counts.
        if tdata is None:
            raise ValueError(f"{self.name}: Point output requires target data")
        results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        values = {}
        dims = {}
        for variable in self.variables:
            values[variable] = model_results.get(variable, tdata.get(variable))
            if values[variable] is None:
                raise KeyError(f"{self.name}: Missing point variable '{variable}'")
            dims[variable] = tdata.dims.get(variable, (FC.STATE, FC.TARGET, FC.TPOINT))
            shape = tuple(tdata.sizes[dimension] for dimension in dims[variable])
            values[variable] = np.broadcast_to(values[variable], shape)
        results[FC.POINT_RESULTS] = dict(
            values=values,
            dims=dims,
            states=mdata[FC.STATE],
            targets=tdata[FC.TARGET],
            tpoints=np.arange(tdata.sizes[FC.TPOINT]),
        )
        return results

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Retain indexed point chunks for deterministic final assembly.

        Parameters
        ----------
        algo
            Calculation algorithm.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total target chunks.
        new_chunk_results
            Pending result-name keyed chunks; this collector consumes its key.
        output_results
            Run-local output buffers.
        """
        # Keep buffers outside model instances so workers share no mutable state.
        output_results.setdefault(FC.POINT_RESULTS, {}).update(
            new_chunk_results.pop(FC.POINT_RESULTS)
        )

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Assemble a dataset with internal state/target/target-point dimensions.

        Parameters
        ----------
        algo
            Calculation algorithm.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total target chunks.
        output_results
            Mapping whose point buffer is replaced with the unified dataset.
        """
        # Concatenate target tiles before joining state slabs in global order.
        chunks = output_results[FC.POINT_RESULTS]
        data_vars = {}
        for variable in self.variables:
            dims = chunks[(0, 0)]["dims"][variable]
            slabs = []
            for state in range(n_chunks_states):
                tiles = [
                    chunks[(state, target)]["values"][variable]
                    for target in range(n_chunks_targets)
                ]
                slabs.append(
                    np.concatenate(tiles, axis=dims.index(FC.TARGET))
                    if FC.TARGET in dims
                    else tiles[0]
                )
            values = (
                np.concatenate(slabs, axis=dims.index(FC.STATE))
                if FC.STATE in dims
                else slabs[0]
            )
            data_vars[variable] = (dims, values)

        # Preserve the public state labels and the original target index.
        coords = {
            FC.TPOINT: chunks[(0, 0)]["tpoints"],
            FC.STATE: np.concatenate(
                [chunks[(state, 0)]["states"] for state in range(n_chunks_states)]
            ),
            FC.TARGET: np.concatenate(
                [chunks[(0, target)]["targets"] for target in range(n_chunks_targets)]
            ),
        }
        output_results[FC.POINT_RESULTS] = Dataset(data_vars=data_vars, coords=coords)
