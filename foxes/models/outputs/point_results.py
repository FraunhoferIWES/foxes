"""Chunk-local point result collection and physical aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output
from foxes.utils import Dict

from ._result_data import _Results, _combine_chunks
from ._aggregation import (
    _finish,
    _finish_states,
    _merge_statistics,
    _point_spatial_statistics,
    _requirements,
    _spatial,
    _state_statistics,
)

if TYPE_CHECKING:
    from foxes.core import Algorithm, FData, MData, TData


class PointResultsOutput(Output):
    """Unify state/target chunks without expanding static input coordinates."""

    def __init__(
        self,
        variables: list[str] | None = None,
        ambient_vars: bool = True,
        waked_vars: bool = True,
        aggregate_points: Literal["all"] | None = None,
        aggregate_states: bool = False,
        result_name: str | None = None,
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
        waked_vars
            Include requested ordinary fields that have ambient counterparts.
        aggregate_points
            ``all`` averages all targets, retaining each target-point position.
            ``None`` retains targets. Point farm/cluster membership is not
            inferred from coordinates and is not supported.
            Dataset attribute ``n_points`` records the input target count.
        aggregate_states
            Use normalized state weights independently per target and target
            point. Workers retain only sufficient statistics for this reduction.
            Dataset attribute ``n_states`` records the selected input count.
        result_name
            Public dataset key, or ``None`` for the aggregation-specific name.
        kwargs
            Common output configuration.

        Raises
        ------
        ValueError
            If ``aggregate_points`` is neither ``None`` nor ``all``, or
            ``aggregate_states`` is not a boolean.
        """
        if aggregate_points not in (None, "all"):
            raise ValueError(
                f"Invalid point aggregation: {aggregate_points!r}; expected None or 'all'"
            )
        if not isinstance(aggregate_states, bool):
            raise ValueError("Point aggregation: aggregate_states must be a boolean")
        self.aggregate_points = aggregate_points
        self.aggregate_states = aggregate_states
        self.result_name = result_name

        # Point collectors read prepared target data and model return values.
        super().__init__(
            mdata_variables=[FC.STATE],
            fdata_variables=[],
            tdata_variables=variables,
            **kwargs,
        )
        self.variables = None if variables is None else list(variables)
        self.ambient_vars = ambient_vars
        self.waked_vars = waked_vars

    @property
    def tdata_variables(self) -> list[str] | None:
        """Target input fields, including reduction dependencies.

        Returns
        -------
        variables
            Selected fields and weights for independent normalized state means;
            ``None`` retains all available fields.
        """
        if not self.aggregate_points and not self.aggregate_states:
            return self.variables
        return _requirements(
            self.variables,
            self.aggregate_points,
            self.aggregate_states,
            self.ambient_vars,
            self.waked_vars,
        )

    @property
    def _collection_key(self) -> str:
        return (
            self.var("point_collection")
            if self.aggregate_points or self.aggregate_states or self.result_name
            else FC.POINT_RESULTS
        )

    @property
    def _public_key(self) -> str:
        if self.result_name is not None:
            return self.result_name
        name = (
            "point_results_aggregate_all" if self.aggregate_points else FC.POINT_RESULTS
        )
        return name + "_mean" if self.aggregate_states else name

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
            State reductions return per-entity sufficient statistics without
            state histories; spatial-only reductions return tile sums/counts.

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
        if self._collection_key not in results:
            raw = self._collect_chunk(mdata, tdata, model_results)
            if self.aggregate_points or self.aggregate_states:
                payload = self._as_result_data({(0, 0): raw}, 1, 1)
                results[self._collection_key] = (
                    _state_statistics(payload)
                    if self.aggregate_states
                    else _point_spatial_statistics(payload)
                )
            else:
                results[self._collection_key] = raw
        return results

    def _collect_chunk(
        self, mdata: MData, tdata: TData, model_results: Any
    ) -> dict[str, Any]:
        """Collect point fields and compact coordinates without changing inputs.

        Parameters
        ----------
        mdata
            Engine-supplied state labels and chunk dimensions.
        tdata
            Engine-supplied target data, read without mutation.
        model_results
            Calculated point fields taking precedence over target data.

        Returns
        -------
        payload
            Selected values, dimensions, labels and singleton-state coordinates
            for static targets. State-dependent coordinates retain every state.
        """
        values, dims = self._field_arrays(tdata, model_results)
        points = np.broadcast_to(
            tdata[FC.TARGETS],
            (mdata.n_states, tdata.n_targets, tdata.n_tpoints, 3),
        )
        if np.array_equal(points, np.broadcast_to(points[:1], points.shape)):
            points = points[:1].copy()
        return dict(
            values=values,
            dims=dims,
            states=mdata[FC.STATE],
            targets=tdata[FC.TARGET],
            tpoints=np.arange(tdata.sizes[FC.TPOINT]),
            points=points,
        )

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
            _requirements(
                [
                    v
                    for v, dimensions in tdata.dims.items()
                    if FC.TARGET in dimensions
                    and v != FC.TARGETS
                    and v not in tdata.sizes
                ],
                self.aggregate_points,
                self.aggregate_states,
                self.ambient_vars,
                self.waked_vars,
            )
            if self.variables is None
            else _requirements(
                self.variables,
                self.aggregate_points,
                self.aggregate_states,
                self.ambient_vars,
                self.waked_vars,
            )
        )
        assert variables is not None
        for variable in variables:
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
        output_results.setdefault(self._collection_key, {}).update(
            new_chunk_results.pop(self._collection_key)
        )

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Assemble the point dataset, retaining axes not selected for reduction.

        Parameters
        ----------
        algo
            Calculation algorithm.
        n_chunks_states
            Total state chunks.
        n_chunks_targets
            Total target chunks.
        output_results
            Mapping whose private point buffer is replaced with the dataset
            under its grouping/state-specific key or ``result_name``. Raw
            collection under ``FC.POINT_RESULTS`` retains internal dimensions
            for algorithm-side public conversion. Other public keys select
            target point zero and rename retained targets to public points.
        """
        # Concatenate target tiles before joining state slabs in global order.
        chunks = output_results.pop(self._collection_key)
        if self.aggregate_states:
            tiles = [
                _finish_states(
                    _merge_statistics([chunks[(s, t)] for s in range(n_chunks_states)])
                )
                for t in range(n_chunks_targets)
            ]
            payload = _combine_chunks(tiles, FC.TARGET)
            if self.aggregate_points:
                payload = _spatial(
                    payload,
                    FC.TARGET,
                    [np.arange(len(payload["coords"][FC.TARGET]))],
                    farm=False,
                )
            payload = self._finish_payload(payload)
        elif self.aggregate_points:
            slabs = [
                _finish_states(
                    _merge_statistics([chunks[(s, t)] for t in range(n_chunks_targets)])
                )
                for s in range(n_chunks_states)
            ]
            payload = self._finish_payload(_combine_chunks(slabs))
        else:
            payload = self._as_result_data(chunks, n_chunks_states, n_chunks_targets)
        dataset = Dataset(
            data_vars=payload["data_vars"],
            coords=payload["coords"],
            attrs=payload["attrs"],
        ).set_coords([FV.X, FV.Y, FV.H])
        if self._public_key != FC.POINT_RESULTS:
            dataset = dataset.sel({FC.TPOINT: 0}, drop=True)
            if FC.TARGET in dataset.dims:
                dataset = dataset.rename({FC.TARGET: FC.POINT})
        output_results[self._public_key] = dataset

    def _finish_payload(self, payload: _Results) -> _Results:
        """Retain geometry while removing private aggregation dependencies."""
        if self.aggregate_points:
            payload["attrs"]["aggregate_points"] = self.aggregate_points
        variables = (
            None
            if self.variables is None
            else [
                v
                for v in self.variables
                if (self.ambient_vars or v not in FV.amb2var)
                and (self.waked_vars or v not in FV.var2amb)
            ]
            + [FV.X, FV.Y, FV.H]
        )
        return _finish(payload, variables, self.aggregate_states)

    def _file_payload(self, payload: _Results) -> _Results:
        """Reduce one complete file interval without changing scientific chunking."""
        if self.aggregate_states:
            if FC.STATE in payload["coords"]:
                payload = _state_statistics(payload)
            payload = _finish_states(payload)
        elif self.aggregate_points and FC.TARGET not in payload["coords"]:
            payload = _finish_states(payload)
        if self.aggregate_points and FC.TARGET in payload["coords"]:
            payload = _spatial(
                payload,
                FC.TARGET,
                [np.arange(len(payload["coords"][FC.TARGET]))],
                farm=False,
            )
        return (
            self._finish_payload(payload)
            if self.aggregate_points or self.aggregate_states
            else payload
        )

    def _as_result_data(
        self,
        chunks: dict[tuple[int, int], Any],
        n_chunks_states: int,
        n_chunks_targets: int,
    ) -> _Results:
        """Assemble NumPy target tiles for collection or buffered file intervals.

        Parameters
        ----------
        chunks
            Complete state/target-indexed payloads in engine-defined chunks.
        n_chunks_states
            Number of state slabs being assembled.
        n_chunks_targets
            Number of target tiles per state slab.

        Returns
        -------
        payload
            Dimensioned fields and geometry, explicit dimension labels and an
            independent attributes mapping; no xarray objects are constructed.

        Raises
        ------
        ValueError
            If engine tiles disagree on fields, dimensions or shared coordinates.
        """
        first = chunks[(0, 0)]
        for (state, target), chunk in chunks.items():
            if (
                chunk["dims"] != first["dims"]
                or chunk["values"].keys() != first["values"].keys()
            ):
                raise ValueError(f"{self.name}: Point fields differ between chunks")
            for name, reference in (
                ("states", chunks[(state, 0)]["states"]),
                ("targets", chunks[(0, target)]["targets"]),
                ("tpoints", first["tpoints"]),
            ):
                if not np.array_equal(
                    chunk[name],
                    reference,
                    equal_nan=chunk[name].dtype.kind in "fcMm",
                ):
                    raise ValueError(f"{self.name}: Point {name} differ between chunks")
        coords = self._assembled_coordinates(chunks, n_chunks_states, n_chunks_targets)
        fields = self._assembled_fields(chunks, n_chunks_states, n_chunks_targets)
        fields.update({v: coords.pop(v) for v in (FV.X, FV.Y, FV.H)})
        return dict(coords=coords, data_vars=fields, attrs={})

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
