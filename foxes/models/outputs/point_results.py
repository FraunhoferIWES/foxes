"""Collect point model results through the engine output lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output
from foxes.utils import Dict

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData


class PointResultsOutput(Output):
    """Unify state/target chunks without expanding static input coordinates."""

    def __init__(
        self,
        variables: list[str] | None = None,
        ambient_vars: bool = True,
        **kwargs: Any,
    ) -> None:
        """Select point fields to collect.

        Parameters
        ----------
        variables
            Point result names, including weights when requested. ``None``
            collects all available state/target fields.
        ambient_vars
            Include ambient point fields such as ``FV.AMB_WS``.
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
        self.variables = None if variables is None else list(variables)
        self.ambient_vars = ambient_vars

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
            Shared mapping with requested point arrays and explicit chunk
            coordinates. Static coordinate payloads have a singleton state axis;
            final assembly restores full public coordinate dimensions.

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
        values, dims = self._field_arrays(tdata, model_results)
        points = np.broadcast_to(
            tdata[FC.TARGETS],
            (mdata.n_states, tdata.n_targets, tdata.n_tpoints, 3),
        )
        if np.array_equal(points, np.broadcast_to(points[:1], points.shape)):
            points = points[:1].copy()
        results[FC.POINT_RESULTS] = dict(
            values=values,
            dims=dims,
            states=mdata[FC.STATE],
            targets=tdata[FC.TARGET],
            tpoints=np.arange(tdata.sizes[FC.TPOINT]),
            points=points,
        )
        return results

    def _field_arrays(
        self, tdata: TData, model_results: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
        """Select requested point fields and expand their declared dimensions.

        Parameters
        ----------
        tdata
            Supplied target data, read without mutation.
        model_results
            Calculated fields taking precedence over values already in ``tdata``.

        Returns
        -------
        values
            Selected arrays, possibly read-only broadcast views of input storage.
        dimensions
            Declared dimensions for each selected field.

        Raises
        ------
        KeyError
            If a selected point field is unavailable.
        """
        values = {}
        dims = {}
        variables = (
            [
                v
                for v, dimensions in tdata.dims.items()
                if FC.TARGET in dimensions and v != FC.TARGETS and v not in tdata.sizes
            ]
            if self.variables is None
            else self.variables
        )
        for variable in variables:
            if not self.ambient_vars and variable in FV.amb2var:
                continue
            values[variable] = model_results.get(variable, tdata.get(variable))
            if values[variable] is None:
                raise KeyError(f"{self.name}: Missing point variable '{variable}'")
            dims[variable] = tdata.dims.get(variable, (FC.STATE, FC.TARGET, FC.TPOINT))
            shape = tuple(tdata.sizes[dimension] for dimension in dims[variable])
            values[variable] = np.broadcast_to(values[variable], shape)
        return values, dims

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
        data_vars = self._assembled_fields(chunks, n_chunks_states, n_chunks_targets)

        # Preserve the public state labels and the original target index.
        coords = self._assembled_coordinates(chunks, n_chunks_states, n_chunks_targets)
        output_results[FC.POINT_RESULTS] = Dataset(data_vars=data_vars, coords=coords)

    def _assembled_coordinates(
        self,
        chunks: dict[tuple[int, int], Any],
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> dict[str, Any]:
        """Restore full public point coordinates from compact worker payloads.

        Parameters
        ----------
        chunks
            State/target-indexed labels and static or state-dependent coordinates.
        n_chunks_states
            Number of state slabs in global order.
        n_chunks_targets
            Number of target tiles per slab.

        Returns
        -------
        coordinates
            State, target and target-point labels plus ``X``, ``Y`` and ``H``
            with dimensions ``(FC.STATE, FC.TARGET, FC.TPOINT)``.
        """
        coords = {
            FC.TPOINT: chunks[(0, 0)]["tpoints"],
            FC.STATE: np.concatenate(
                [chunks[(state, 0)]["states"] for state in range(n_chunks_states)]
            ),
            FC.TARGET: np.concatenate(
                [chunks[(0, target)]["targets"] for target in range(n_chunks_targets)]
            ),
        }
        points = np.concatenate(
            [
                np.concatenate(
                    [
                        np.broadcast_to(
                            chunks[(s, t)]["points"],
                            (
                                len(chunks[(s, t)]["states"]),
                                *chunks[(s, t)]["points"].shape[1:],
                            ),
                        )
                        for t in range(n_chunks_targets)
                    ],
                    axis=1,
                )
                for s in range(n_chunks_states)
            ],
            axis=0,
        )
        for i, variable in enumerate((FV.X, FV.Y, FV.H)):
            coords[variable] = ((FC.STATE, FC.TARGET, FC.TPOINT), points[..., i])
        return coords

    def _assembled_fields(
        self,
        chunks: dict[tuple[int, int], Any],
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> dict[str, Any]:
        """Join target tiles before state slabs along each field's declared axes.

        Parameters
        ----------
        chunks
            Complete state/target-indexed point payloads with matching schemas.
        n_chunks_states
            Number of state slabs in global order.
        n_chunks_targets
            Number of target tiles per slab.

        Returns
        -------
        fields
            Dimensioned arrays suitable for dataset construction. Static axes
            use the first slab/tile rather than duplicating their values.
        """
        data_vars = {}
        for variable in chunks[(0, 0)]["values"]:
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
        return data_vars
