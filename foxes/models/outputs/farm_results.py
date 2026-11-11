"""Chunk-local farm result collection and NetCDF output models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output, get_engine
from foxes.utils import Dict

from ._farm_result_data import (
    _FarmResults,
    _combine_chunks,
    _create_dataset,
    _data_to_nc,
    _select_states,
)

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
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
        kwargs
            Additional parameters for :class:`foxes.core.Output`.
        """
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

    def _collect_chunk(self, mdata: MData, fdata: FData) -> _FarmResults:
        """Collect selected farm fields with explicit state/turbine coordinates.

        Parameters
        ----------
        mdata
            Chunk metadata supplying state coordinates when available.
        fdata
            Farm-ordered arrays, read without mutation.

        Returns
        -------
        payload
            Coordinates, selected ``(state, turbine)`` fields and an empty
            attributes mapping. Broadcast fields may share the input storage.

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
        variables = (
            list(fdata) if self.fdata_variables is None else self.fdata_variables
        )
        # Filter physical variables without changing chunk-local weight dimensions.
        for variable in variables:
            if variable in _FARM_COORDINATES:
                continue
            if not self.ambient_vars and variable in FV.amb2var:
                continue
            if not self.waked_vars and variable in FV.var2amb:
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
        return dict(coords=coords, data_vars=data_vars, attrs={})

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
            The shared mapping with a NumPy payload under ``FC.FARM_RESULTS``.

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
        if mdata.chunki_points in (None, 0) and FC.FARM_RESULTS not in chunk_results:
            chunk_results[FC.FARM_RESULTS] = self._collect_chunk(mdata, fdata)
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
            Shared pending payloads. This output consumes ``FC.FARM_RESULTS``.
        output_results
            The result mapping updated with indexed farm payloads.
        """
        # Consume only the dataset output's pending result key.
        chunks: Dict[tuple[int, int], Any] = new_chunk_results.pop(
            FC.FARM_RESULTS, Dict()
        )
        output_results.setdefault(FC.FARM_RESULTS, Dict()).update(chunks)

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
            Updated in place with the dataset under ``FC.FARM_RESULTS``.
            State labels and turbine coordinates are retained. Uniform weights
            have dimensions ``(FC.STATE,)``; other fields have dimensions
            ``(FC.STATE, FC.TURBINE)``.

        Raises
        ------
        ValueError
            If target chunks or farm payload schemas are incompatible.
        """
        # Assemble chunks in global state order rather than completion order.
        chunks = output_results[FC.FARM_RESULTS]
        payload = _combine_chunks(
            [chunks[(index, 0)] for index in range(n_chunks_states)]
        )
        output_results[FC.FARM_RESULTS] = _create_dataset(
            payload, *self._metadata(algo)
        )


class FarmResultsCountWriter(FarmResultsOutput):
    """Farm files with a fixed state limit, independent of engine chunk sizes."""

    def __init__(
        self,
        base_name: str | Path,
        n_states_per_file: int,
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure count-limited farm files.

        Parameters
        ----------
        base_name
            File prefix; filenames append ``_<file index:06d>.nc``.
        n_states_per_file
            Positive integer limit. Every file except the last has this many states.
        write_nc_pars
            Keyword arguments forwarded to ``foxes.utils.write_nc`` for every file.
            Verbosity defaults to the executing algorithm's level.
        kwargs
            Farm selection, metadata and path options for ``FarmResultsOutput``.

        Raises
        ------
        ValueError
            If ``n_states_per_file`` is not a positive integer.
        """
        # Validate the public file limit before initializing model state.
        if (
            isinstance(n_states_per_file, bool)
            or not isinstance(n_states_per_file, (int, np.integer))
            or n_states_per_file <= 0
        ):
            raise ValueError("n_states_per_file must be a positive integer")
        self.base_name = base_name
        self.n_states_per_file = n_states_per_file
        self.write_nc_pars = {} if write_nc_pars is None else dict(write_nc_pars)
        super().__init__(**kwargs)

        # Keep writer buffers separate from the in-memory farm dataset output.
        self.FARM_RESULTS = self.var(FC.FARM_RESULTS)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect an independent farm payload for this writer.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            Chunk metadata and state coordinates.
        fdata
            Farm-ordered arrays selected for this writer.
        tdata
            Unused target data; may be ``None``.
        model_results
            Unused model return values.
        chunk_results
            Shared payloads to update, or ``None`` to create a mapping.

        Returns
        -------
        chunk_results
            The shared mapping with this model's private farm payload key.
        """
        # Collect independent variables even when another farm output ran first.
        chunk_results = Output.calculate(
            self, algo, mdata, fdata, tdata, model_results, chunk_results
        )
        if mdata.chunki_points != 0:
            return chunk_results
        payload = self._collect_chunk(mdata, fdata)
        payload["attrs"].update(base_name=str(self.base_name))
        chunk_results[self.FARM_RESULTS] = payload
        return chunk_results

    def _ready(self, chunks: list[_FarmResults]) -> bool:
        """Check whether enough states are buffered for a complete file."""
        # Count buffered states without concatenating physical fields.
        return (
            sum(len(chunk["coords"][FC.STATE]) for chunk in chunks)
            >= self.n_states_per_file
        )

    def _file_size(self, states: np.ndarray) -> int:
        """Select at most the configured number of states."""
        # Apply the limit to final chunks as well as intermediate chunks.
        return min(self.n_states_per_file, len(states))

    def _get_fpath(self, attrs: dict[str, Any]) -> Path:
        """Resolve the next count-limited filename."""
        # Apply the standard output-directory and filename transformation.
        return self.get_fpath(f"{self.base_name}_{attrs['index']:06d}.nc")

    def _flush(self, algo: Algorithm, results: dict[str, Any]) -> None:
        """Submit complete files and retain only the incomplete state interval.

        Parameters
        ----------
        algo
            Algorithm supplying requested turbine metadata.
        results
            Run-local writer buffer, updated in place with unfinished chunks,
            submitted futures and the next file index. ``last`` flushes the
            final incomplete interval.

        Raises
        ------
        RuntimeError
            If no engine is active to submit file writes.
        """
        # Combine buffered chunks only when a file is ready or the run ends.
        engine = get_engine(error=False)
        if engine is None:
            raise RuntimeError(f"{self.name}: File writing requires an active engine")
        payload = _combine_chunks(results["chunks"])
        n_states = len(payload["coords"][FC.STATE])
        start = 0
        while start < n_states:
            remaining = payload["coords"][FC.STATE][start:]
            size = self._file_size(remaining)
            if (
                not results["last"]
                and size == len(remaining)
                and not self._ready([_select_states(payload, start, n_states)])
            ):
                break

            # Submit an independent selection with accurate per-file metadata.
            selected = _select_states(payload, start, start + size)
            selected["attrs"].update(index=results["index"], n_states=size)
            self._file_attrs(selected)
            future = engine.submit(
                _data_to_nc,
                self._get_fpath(selected["attrs"]),
                selected,
                *self._metadata(algo),
                {"verbosity": algo.verbosity, **self.write_nc_pars},
            )
            results["futures"].append(future)
            results["index"] += 1
            start += size

        # Retain the unfinished period or count interval for the next chunk.
        results["chunks"] = (
            [_select_states(payload, start, n_states)] if start < n_states else []
        )

    def _file_attrs(self, payload: _FarmResults) -> None:
        """Attach writer-specific file attributes to a selected state interval."""
        # Record the configured count independently of the final file's size.
        payload["attrs"]["n_states_per_file"] = self.n_states_per_file

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Buffer ordered chunks and submit every complete file interval.

        Parameters
        ----------
        algo
            The calculation algorithm and farm metadata.
        n_chunks_states
            The total number of state chunks; the last chunk flushes the remainder.
        n_chunks_targets
            Number of point chunks. Farm data is collected from the first only.
        new_chunk_results
            Shared pending payloads. Only this writer's private key is consumed.
        output_results
            Updated with private buffers and write futures, not a full dataset.

        Raises
        ------
        ValueError
            If state chunks arrive out of order.
        """
        # Require ordered farm chunks and initialize run-local writer buffers.
        chunks: Dict[tuple[int, int], Any] = new_chunk_results.pop(
            self.FARM_RESULTS, Dict()
        )
        results = output_results.setdefault(
            self.FARM_RESULTS,
            dict(
                chunks=[],
                index=0,
                next_chunk=0,
                futures=[],
                last=False,
                last_state=None,
            ),
        )

        # Emit all complete intervals, including multiple files in the last chunk.
        for key, chunk in chunks.items():
            expected = (results["next_chunk"], 0)
            if key != expected:
                raise ValueError(
                    f"{self.name}: Unexpected chunk {key}, expected {expected}"
                )
            self._validate_chunk(chunk, results["last_state"])
            results["last_state"] = chunk["coords"][FC.STATE][-1]
            results["chunks"].append(chunk)
            results["next_chunk"] += 1
            results["last"] = results["next_chunk"] == n_chunks_states
            if self._ready(results["chunks"]) or results["last"]:
                self._flush(algo, results)

    def _validate_chunk(self, chunk: _FarmResults, previous: np.generic | None) -> None:
        """Accept state coordinates for count-based files without extra restrictions.

        Parameters
        ----------
        chunk
            Farm payload about to enter the writer's run-local buffer.
        previous
            Last coordinate from the preceding chunk, or ``None`` initially.
            Calendar-policy subclasses validate cross-chunk ordering here.
        """

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Wait for all submitted files and remove this writer's private buffers.

        Parameters
        ----------
        algo
            The calculation algorithm.
        n_chunks_states
            Expected number of processed state chunks.
        n_chunks_targets
            The total number of target chunks.
        output_results
            Shared results; other outputs' values are left untouched.

        Raises
        ------
        ValueError
            If not all expected state chunks were delivered.

        Notes
        -----
        Worker write errors propagate when their futures are awaited.
        """
        # Ensure every state chunk was handled before resolving file writes.
        results = output_results.pop(self.FARM_RESULTS)
        if results["next_chunk"] != n_chunks_states:
            raise ValueError(f"{self.name}: Missing farm result chunks")

        # Resolve worker failures before reporting that file output has finished.
        engine = get_engine(error=False)
        if engine is None:
            raise RuntimeError(f"{self.name}: File writing requires an active engine")
        engine.await_results(results["futures"])
