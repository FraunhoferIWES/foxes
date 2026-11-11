"""Chunk-local farm result collection and physical aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output
from foxes.utils import Dict
from ._aggregation import (
    _farm_groups,
    _finish,
    _finish_states,
    _merge_statistics,
    _requirements,
    _spatial,
    _state_statistics,
)

from ._result_data import (
    _Results,
    _combine_chunks,
    _create_dataset,
)

if TYPE_CHECKING:
    from foxes.core import Algorithm
    from foxes.core.data import FData, MData, TData


_FARM_COORDINATES = frozenset(
    {FC.STATE, FC.TURBINE, FC.FARM, FC.CLUSTER, FC.XYH, FV.TXYH, FC.TNAME}
)


class FarmResultsOutput(Output):
    """Farm results collected from NumPy chunks into one xarray dataset."""

    def __init__(
        self,
        variables: list[str] | None = None,
        ambient_vars: bool = True,
        waked_vars: bool = True,
        add_turbine_names: bool = True,
        add_farm_names: bool = True,
        add_cluster_names: bool = True,
        aggregate_turbines: Literal["wind_farm", "cluster", "all"] | None = None,
        aggregate_states: bool = False,
        result_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the farm result selection and metadata.

        Parameters
        ----------
        variables
            Farm variables to collect, or ``None`` for all available fields.
            Physical fields must have dimensions ``(FC.STATE, FC.TURBINE)``.
        ambient_vars
            Include requested ambient fields such as ``FV.AMB_P``.
        waked_vars
            Include requested ordinary fields that have ambient counterparts.
        add_turbine_names
            Attach turbine names along ``FC.TURBINE``.
        add_farm_names
            Attach wind-farm names when the farm contains named memberships.
        add_cluster_names
            Attach cluster names when the farm contains named memberships.
        aggregate_turbines
            Group turbines by wind farm, cluster, or all turbines. ``None``
            retains individual turbines. Unassigned memberships form an explicit
            ``<unassigned>`` group. Extensive fields are summed, intensive
            fields averaged and directions circularly averaged. Ratios are
            recomputed from reduced power; mixed native power units are rejected.
            Dataset attribute ``n_turbines`` records the input turbine count.
        aggregate_states
            Reduce states using weights normalized independently per turbine,
            before grouping turbines. Workers return sufficient statistics,
            not full state histories. Zero-weight samples do not propagate NaNs.
            Dataset attribute ``n_states`` records the selected input count.
        result_name
            Public dataset key, or ``None`` for the grouping-specific name.
        kwargs
            Additional parameters for :class:`foxes.core.Output`.

        Raises
        ------
        ValueError
            If ``aggregate_turbines`` is not a supported grouping or
            ``aggregate_states`` is not a boolean.
        """
        if aggregate_turbines not in (None, "wind_farm", "cluster", "all"):
            raise ValueError(f"Invalid farm aggregation: {aggregate_turbines!r}")
        if not isinstance(aggregate_states, bool):
            raise ValueError("Farm aggregation: aggregate_states must be a boolean")
        self.aggregate_turbines = aggregate_turbines
        self.aggregate_states = aggregate_states
        self.result_name = result_name

        # Declare the chunk input requirements and retain selection flags.
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

    @property
    def fdata_variables(self) -> list[str] | None:
        """Farm input fields, including reduction dependencies.

        Returns
        -------
        variables
            Selected fields, required power inputs and per-turbine state weights;
            ``None`` retains all available fields.
            Without aggregation, the configured list remains directly editable
            before initialization.
        """
        if not self.aggregate_turbines and not self.aggregate_states:
            return super().fdata_variables
        return _requirements(
            super().fdata_variables,
            self.aggregate_turbines,
            self.aggregate_states,
            self.ambient_vars,
            self.waked_vars,
        )

    @property
    def _collection_key(self) -> str:
        return (
            self.var("farm_collection")
            if self.aggregate_turbines or self.aggregate_states or self.result_name
            else FC.FARM_RESULTS
        )

    @property
    def _public_key(self) -> str:
        if self.result_name is not None:
            return self.result_name
        name = (
            f"farm_results_aggregate_{self.aggregate_turbines}"
            if self.aggregate_turbines
            else FC.FARM_RESULTS
        )
        return name + "_mean" if self.aggregate_states else name

    def _collect_chunk(
        self, algo: Algorithm, mdata: MData, fdata: FData, reduce_states: bool = True
    ) -> _Results:
        """Collect selected farm fields with explicit state/turbine coordinates.

        Parameters
        ----------
        algo
            Algorithm supplying original-order group and turbine-type metadata.
        mdata
            Chunk metadata supplying state coordinates when available.
        fdata
            Farm-ordered arrays, read without mutation.
        reduce_states
            Reduce state histories now; interval writers defer reduction until
            this worker's samples have been partitioned at exact file boundaries.

        Returns
        -------
        payload
            Selected coordinates, fields and attributes. Raw fields have
            ``(state, turbine)`` dimensions and may share input storage;
            requested spatial/state reductions remove the corresponding axes.

        Raises
        ------
        KeyError
            If a requested physical field is missing.
        ValueError
            If a selected field has dimensions other than state/turbine.
        """
        # Retain dimensions even when all physical fields are filtered out.
        coords = {
            FC.STATE: mdata[FC.STATE] if FC.STATE in mdata else fdata[FC.STATE],
            FC.TURBINE: fdata[FC.TURBINE],
        }
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        selection = super().fdata_variables
        variables = _requirements(
            list(fdata) if selection is None else selection,
            self.aggregate_turbines,
            self.aggregate_states,
            self.ambient_vars,
            self.waked_vars,
        )
        assert variables is not None
        # Filter physical variables without changing chunk-local weight dimensions.
        for variable in variables:
            if variable in _FARM_COORDINATES:
                continue
            if variable not in fdata:
                raise KeyError(
                    f"{self.name}: Missing requested farm variable '{variable}'"
                )
            dims = fdata.dims[variable]
            if dims != (FC.STATE, FC.TURBINE):
                raise ValueError(
                    f"{self.name}: Expecting dimensions {(FC.STATE, FC.TURBINE)} "
                    f"for '{variable}', got {dims}"
                )
            # Expand broadcast farm fields before creating dimensioned output arrays.
            shape = (len(coords[FC.STATE]), len(coords[FC.TURBINE]))
            data_vars[variable] = (dims, np.broadcast_to(fdata[variable], shape))
        payload: _Results = dict(coords=coords, data_vars=data_vars, attrs={})
        if self.aggregate_states and reduce_states:
            return _state_statistics(payload)
        if self.aggregate_turbines and not self.aggregate_states:
            return self._file_payload(algo, payload)
        return payload

    def _file_payload(self, algo: Algorithm, payload: _Results) -> _Results:
        """Finish one collector or file interval using the same physical reductions."""
        if not self.aggregate_turbines and not self.aggregate_states:
            return payload
        if self.aggregate_states:
            if FC.STATE in payload["coords"]:
                payload = _state_statistics(payload)
            payload = _finish_states(payload)
        if self.aggregate_turbines:
            groups, dimension, labels = _farm_groups(
                algo,
                self.aggregate_turbines,
                payload["coords"][FC.TURBINE],
                payload["data_vars"],
            )
            payload = _spatial(payload, FC.TURBINE, groups, dimension, labels)
            payload["attrs"]["aggregate_turbines"] = self.aggregate_turbines
        variables = _requirements(
            super().fdata_variables, None, False, self.ambient_vars, self.waked_vars
        )
        return _finish(payload, variables, self.aggregate_states)

    def _metadata(
        self, algo: Algorithm
    ) -> tuple[
        Sequence[str] | None, Sequence[str | None] | None, Sequence[str | None] | None
    ]:
        """Select farm metadata consistently for datasets and files.

        Parameters
        ----------
        algo
            Algorithm supplying metadata in original turbine order.

        Returns
        -------
        turbine_names
            Turbine names, or ``None`` when disabled.
        wind_farm_names
            Wind-farm memberships, or ``None`` when disabled.
        cluster_names
            Cluster memberships, or ``None`` when disabled.
        """
        # Resolve metadata only when the corresponding output flag is enabled.
        return (
            algo.farm.turbine_names if self.add_turbine_names else None,
            algo.farm.wind_farm_list if self.add_farm_names else None,
            algo.farm.cluster_list if self.add_cluster_names else None,
        )

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect the farm dataset payload for one chunk.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            Chunk metadata and state coordinates.
        fdata
            Farm-ordered data with leading state and turbine dimensions.
        tdata
            Target data, unused by this output; may be ``None``.
        model_results
            Model return values, unused because arrays are read from ``fdata``.
        chunk_results
            Shared output payloads to update, or ``None`` to create a mapping.

        Returns
        -------
        chunk_results
            Shared NumPy payloads under ``FC.FARM_RESULTS`` for raw collection,
            or a model-private key for reduced/custom-named collection.

        Raises
        ------
        KeyError
            If a requested farm variable is unavailable.
        ValueError
            If a selected physical variable does not have state/turbine dimensions.
        """
        # Add the dataset payload without discarding other outputs' results.
        chunk_results = super().calculate(
            algo, mdata, fdata, tdata, model_results, chunk_results
        )
        if (
            mdata.chunki_points in (None, 0)
            and self._collection_key not in chunk_results
        ):
            chunk_results[self._collection_key] = self._collect_chunk(
                algo, mdata, fdata
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
        """Retain new farm payloads by chunk index for final state-order assembly.

        Parameters
        ----------
        algo
            The calculation algorithm.
        n_chunks_states
            The total number of state chunks.
        n_chunks_targets
            The total number of target chunks.
        new_chunk_results
            Shared pending payloads. This output consumes only its collection key.
        output_results
            The result mapping updated with indexed farm payloads.
        """
        # Consume only the dataset output's pending result key.
        chunks: Dict[tuple[int, int], Any] = new_chunk_results.pop(
            self._collection_key, Dict()
        )
        output_results.setdefault(self._collection_key, Dict()).update(chunks)

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Replace collected farm payloads with one state-ordered dataset.

        Parameters
        ----------
        algo
            The algorithm supplying turbine, farm and cluster metadata.
        n_chunks_states
            The total number of state chunks.
        n_chunks_targets
            Number of point chunks. Farm data is collected from the first only.
        output_results
            Updated with the dataset under its grouping/state-specific key or
            ``result_name``. Unaggregated fields retain state/turbine dimensions
            and compress uniform weights to ``(FC.STATE,)``. State reduction
            removes states; spatial grouping replaces turbines with farm/cluster
            groups or removes that axis for ``all``.

        Raises
        ------
        ValueError
            If target chunks or farm payload schemas are incompatible.
        """
        # Assemble chunks in global state order rather than completion order.
        chunks = output_results.pop(self._collection_key)
        ordered = [chunks[(index, 0)] for index in range(n_chunks_states)]
        payload = (
            _merge_statistics(ordered)
            if self.aggregate_states
            else _combine_chunks(ordered)
        )
        if self.aggregate_states:
            payload = self._file_payload(algo, payload)
        output_results[self._public_key] = _create_dataset(
            payload, *self._metadata(algo)
        )
