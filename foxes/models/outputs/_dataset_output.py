"""Shared farm-dataset collection for constructor-defined output operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
from foxes.core import get_engine
from foxes.utils import Dict

from ._aggregation import _merge_statistics
from ._result_data import _combine_chunks
from ._units import _normalize_farm_power, _set_units
from .farm_results import FarmResultsOutput
from .point_results import PointResultsOutput

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, FData, MData, Model, PointDataModelList, TData
    from foxes.core.model import LoadedData


def _farm_fields(
    fdata: FData, requested: tuple[str, ...] | None, previous: Dataset | None
) -> dict[str, Any]:
    """Copy selected farm fields while restoring precomputed dimensions.

    Parameters
    ----------
    fdata
        Chunk-local farm data, which may broadcast static fields over states.
    requested
        Selected variable names, or ``None`` for all available scientific fields.
    previous
        Precomputed dataset defining original scalar/state/turbine dimensions,
        or ``None`` when fields were calculated by this run.

    Returns
    -------
    fields
        Variable-to-``(dimensions, copied values)`` mapping without coordinate
        entries. Scalar and single-axis precomputed fields are not expanded.
    """
    fields = {}
    for variable, dims in fdata.dims.items():
        if variable in fdata.sizes:
            continue
        if requested is not None and variable not in requested:
            continue
        if previous is not None and variable in previous:
            original = previous[variable].dims
            if original == (FC.TURBINE,) and dims == (FC.STATE, FC.TURBINE):
                fields[variable] = (original, fdata[variable][0].copy())
                continue
            if original == (FC.STATE,) and dims == (FC.STATE, FC.TURBINE):
                fields[variable] = (original, fdata[variable][:, 0].copy())
                continue
            if original == ():
                fields[variable] = (original, fdata[variable][0, 0].copy())
                continue
        if dims in ((FC.STATE,), (FC.TURBINE,), (FC.STATE, FC.TURBINE)):
            fields[variable] = (
                dims,
                np.broadcast_to(
                    fdata[variable], tuple(fdata.sizes[d] for d in dims)
                ).copy(),
            )
    return fields


def _join_point_blocks(blocks: list[dict[str, Any]], n_points: int) -> dict[str, Any]:
    """Join point slices while keeping static coordinates compact when possible.

    Parameters
    ----------
    blocks
        Nonempty point-ordered collector payloads for a single state slab.
    n_points
        Number of requested targets, defining their assembled global indices.

    Returns
    -------
    payload
        Concatenated target-axis fields and coordinates. Static coordinate
        blocks expand only if another block has state-dependent coordinates.
    """
    first = blocks[0]
    point_states = max(block["points"].shape[0] for block in blocks)
    return {
        **first,
        "values": {
            v: np.concatenate([b["values"][v] for b in blocks], axis=1)
            for v in first["values"]
        },
        "points": np.concatenate(
            [
                np.broadcast_to(b["points"], (point_states, *b["points"].shape[1:]))
                for b in blocks
            ],
            axis=1,
        ),
        "targets": np.arange(n_points),
    }


class DatasetOutput(FarmResultsOutput):
    """Collect farm chunks under a private key before a final output operation."""

    def __init__(
        self, result_name: str | None = None, use_points: bool = False, **kwargs: Any
    ) -> None:
        """Configure private farm or point collection without retaining datasets.

        Parameters
        ----------
        result_name
            Public result key; ``None`` uses this output's model name.
        use_points
            Collect supplied or configured calculation targets instead of farm fields.
        kwargs
            Farm-result selection and common output constructor options.
        """
        # Retain only selection and naming configuration, never result buffers.
        super().__init__(**kwargs)
        if use_points:
            self._mdata_variables = None
            self._tdata_variables = None
        self.result_name = result_name or self.name
        self.use_points = use_points
        self._collector = PointResultsOutput() if use_points else None
        self._point_models: PointDataModelList | None = None
        self._point_parameters: list[dict[str, Any]] = []
        self._chunk_size_points: int | None = None

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """Prepare point dependencies only for outputs that consume point fields.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        loaded_data
            Existing initialization data to extend, or ``None`` to create it.
        force
            Reinitialize an already initialized output.
        verbosity
            Diagnostic verbosity; zero is silent.

        Returns
        -------
        loaded_data
            Initialized output and dependency data.
        """
        # Scientific dependencies are prepared only when this output is selected.
        if self.use_points and (not self.initialized or force):
            self._point_models, self._point_parameters = algo._collect_point_models()
        return super().initialize(algo, loaded_data, force, verbosity)

    def sub_models(self) -> list[Model]:
        """Return point models and collectors governed by this output's lifecycle.

        Returns
        -------
        models
            Configured point dependencies, excluding unset entries.
        """
        return [
            model
            for model in (self._point_models, self._collector)
            if model is not None
        ]

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Snapshot the point-chunk limit before entering read-only execution.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        data_stash
            Lifecycle-owned stash for temporarily removed configuration, or ``None``.
        sel
            Engine selection applied by the base output lifecycle.
        verbosity
            Diagnostic verbosity; zero is silent.
        """
        # Capture the scalar point limit before entering read-only running state.
        engine = get_engine(error=False)
        self._chunk_size_points = None if engine is None else engine.chunk_size_points
        super().set_running(algo, data_stash, sel, verbosity)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect selected fields once per state slab or supplied target tile.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        mdata
            Chunk metadata with state coordinates and loaded calculation grids.
        fdata
            Farm fields with leading dimensions ``(FC.STATE, FC.TURBINE)``.
        tdata
            Supplied target data, or ``None`` for a farm-only calculation.
        model_results
            Scientific calculation returns; fields are read from the data containers.
        chunk_results
            Shared chunk-result mapping to extend, or ``None`` to create it.

        Returns
        -------
        chunk_results
            The extended mapping, containing a private farm payload or point payload.
            Farm fields preserve original scalar, turbine-only and state dimensions;
            repeated farm fields in later point tiles are not transported again.
        """
        # Farm fields repeat across point tiles; collect each state slab once.
        results = Dict() if chunk_results is None else chunk_results
        if self.use_points:
            results[self.var("data")] = self._collect_points(algo, mdata, fdata, tdata)
        elif mdata.chunki_points in (None, 0):
            requested = self._farm_variables()
            if requested == ():
                return results
            if self.aggregate_turbines or self.aggregate_states:
                results[self.var("data")] = self._collect_chunk(algo, mdata, fdata)
                return results
            previous = algo.prev_farm_results(mdata=mdata)
            results[self.var("data")] = {
                "coords": {FC.STATE: mdata[FC.STATE], FC.TURBINE: fdata[FC.TURBINE]},
                "data_vars": _farm_fields(fdata, requested, previous),
                "attrs": {},
            }
            _normalize_farm_power(results[self.var("data")], algo, previous)
        return results

    def _farm_variables(self) -> tuple[str, ...] | None:
        """Select scientific fields transported by this consumer.

        Returns
        -------
        variables
            Farm field names, an empty tuple for metadata-only results, or
            ``None`` for all available fields. Collection preserves declared
            scalar, turbine-only and state-dependent dimensions.
        """
        return None

    def _collect_points(
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData | None
    ) -> dict[str, Any]:
        """Collect supplied targets or evaluate configured points in bounded blocks.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        mdata
            Chunk metadata with state coordinates and loaded calculation grids.
        fdata
            Farm fields with leading dimensions ``(FC.STATE, FC.TURBINE)``.
        tdata
            Supplied target data, or ``None`` for a farm-only calculation.

        Returns
        -------
        payload
            Collected state/target fields and coordinates. Static coordinates retain
            one state row; dynamic coordinates retain their state axis.

        Raises
        ------
        ValueError
            If neither supplied targets nor configured points are available.
        RuntimeError
            If the point collector or scientific dependencies were not initialized.
        """
        from ._point_calculation import point_blocks

        # Supplied targets take precedence; farm-only runs evaluate configured points.
        if self._collector is None:
            raise RuntimeError(f"{self.name}: No point collector configured")
        if tdata is not None:
            return self._collector.calculate(algo, mdata, fdata, tdata, {})[
                FC.POINT_RESULTS
            ]
        points = self.get_calc_points_from_chunk(mdata)
        if points is None:
            raise ValueError(f"{self.name}: Point output requires supplied target data")
        if self._point_models is None:
            raise RuntimeError(
                f"{self.name}: Initialize point models before calculation"
            )
        blocks = [
            self._collector.calculate(algo, mdata, fdata, block, {})[FC.POINT_RESULTS]
            for block in point_blocks(
                algo,
                mdata,
                fdata,
                points,
                self._point_models,
                self._point_parameters,
                self._chunk_size_points,
            )
        ]
        return _join_point_blocks(blocks, points.shape[-2])

    def get_calc_points_from_chunk(self, mdata: MData) -> np.ndarray | None:
        """Return configured points available during farm-chunk evaluation.

        Parameters
        ----------
        mdata
            Chunk metadata and loaded extra data containing calculation grids.

        Returns
        -------
        points
            Static or state-dependent coordinates, or ``None`` for supplied
            targets only. Implementations must not retain arrays on the model.
        """
        return None

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Transfer private chunk payloads into engine-owned accumulation buffers.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        n_chunks_states
            Total engine state chunks.
        n_chunks_targets
            Total engine target chunks.
        new_chunk_results
            New chunk payloads indexed by state/target chunk; the private entry is consumed.
        output_results
            Run-local mapping that owns output buffers and final public results.
            Other outputs' entries must remain untouched.
        """
        # Keep buffers in the engine-owned mapping, separate from returned values.
        key = self.var("data")
        output_results.setdefault(key, {}).update(new_chunk_results.pop(key, {}))

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Assemble a parent-side dataset and apply the configured final operation.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        n_chunks_states
            Total number of engine state chunks.
        n_chunks_targets
            Total number of engine target chunks.
        output_results
            Run-local mapping that owns output buffers and final public results.
            Other outputs' entries must remain untouched.

        Notes
        -----
        The private buffer is consumed. The public result is stored under
        ``result_name``; scientific arrays and datasets are never attached to the
        output model. Point data uses ``(FC.STATE, FC.POINT)`` dimensions.
        """
        from xarray import Dataset

        # Construct xarray only on the parent, after all scientific chunks finish.
        chunks = output_results.pop(self.var("data"), {})
        if self.use_points and chunks:
            if self._collector is None:
                raise RuntimeError(f"{self.name}: No point collector configured")
            collected: Dict[str, Any] = Dict({FC.POINT_RESULTS: chunks})
            self._collector.finalize_results(
                algo, n_chunks_states, n_chunks_targets, collected
            )
            data = (
                collected[FC.POINT_RESULTS]
                .sel({FC.TPOINT: 0}, drop=True)
                .rename({FC.TARGET: FC.POINT})
            )
        elif chunks:
            ordered = [chunks[(s, 0)] for s in range(n_chunks_states)]
            payload = (
                _merge_statistics(ordered)
                if self.aggregate_states
                else _combine_chunks(ordered)
            )
            if self.aggregate_states:
                payload = self._file_payload(algo, payload)
            data = Dataset(**payload)
        else:
            data = Dataset()

        _set_units(data)
        output_results[self.result_name] = self.process(algo, data, output_results)

    def process(
        self, algo: Algorithm, data: Dataset, output_results: Dict[str, Any]
    ) -> Any:
        """Transform a collected dataset into the configured public output.

        Parameters
        ----------
        algo
            Algorithm supplying scientific models, farm metadata and verbosity.
        data
            Parent-side farm or point dataset, borrowed only for this operation.
        output_results
            Run-owned initial results. Concrete consumers explicitly retrieve
            and remove their own rendering or formatting inputs.

        Returns
        -------
        result
            Public output value stored under ``result_name``.

        Raises
        ------
        NotImplementedError
            If a concrete consumer does not implement its final transformation.
        """
        raise NotImplementedError
