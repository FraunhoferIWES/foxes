"""Engine-chunk and calendar-period NetCDF point writer policies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import Output
from foxes.utils import Dict, write_nc

from ._result_writers import _ResultsCountWriter, _ResultsPeriodWriter
from ._result_data import _Results, _combine_chunks
from ._aggregation import _merge_statistics, _point_spatial_statistics
from .point_results import PointResultsOutput

if TYPE_CHECKING:
    from foxes.core import Algorithm, Engine, FData, MData, TData


def _point_data_to_nc(
    fpath: Path, payload: _Results, write_nc_pars: dict[str, Any]
) -> None:
    """Write selected point fields with every retained target-point position."""
    dataset = Dataset(
        data_vars=payload["data_vars"], coords=payload["coords"], attrs=payload["attrs"]
    ).set_coords([FV.X, FV.Y, FV.H])
    write_nc(ds=dataset, fpath=fpath, **write_nc_pars)


class PointResultsCountWriter(_ResultsCountWriter, PointResultsOutput):
    """Point files with a fixed state limit, containing every engine target tile.

    Unaggregated files retain ``(state, target, target_point)`` coordinates and fields,
    including every target point rather than selecting target point zero.
    """

    _RESULTS = FC.POINT_RESULTS

    def __init__(
        self,
        base_name: str | Path,
        n_states_per_file: int,
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure count-limited point files.

        Parameters
        ----------
        base_name
            File prefix; filenames append ``_<file index:06d>.nc``.
        n_states_per_file
            Positive integer state limit, except for the final file.
        write_nc_pars
            Options for ``foxes.utils.write_nc``, inheriting algorithm verbosity.
        kwargs
            Field selection, ``aggregate_points``, ``aggregate_states`` and
            path options for ``PointResultsOutput``.

        Raises
        ------
        ValueError
            If the state limit is not a positive integer.
        """
        super().__init__(
            base_name, n_states_per_file, write_nc_pars=write_nc_pars, **kwargs
        )
        self._tiles_key = self.var("point_tiles")

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Collect an independent payload from the engine's target-data chunk.

        Parameters
        ----------
        algo
            Executing algorithm.
        mdata
            Engine state and target chunk metadata.
        fdata
            Farm fields, unused by this point writer.
        tdata
            Required target data, read without mutation.
        model_results
            Calculated point fields.
        chunk_results
            Shared output mapping or ``None``.

        Returns
        -------
        chunk_results
            Shared results with this writer's private point payload.

        Raises
        ------
        ValueError
            If target data is missing.
        """
        if tdata is None:
            raise ValueError(f"{self.name}: Point output requires target data")
        results = Output.calculate(
            self, algo, mdata, fdata, tdata, model_results, chunk_results
        )
        raw = self._collect_chunk(mdata, tdata, model_results)
        if self.aggregate_points or self.aggregate_states:
            payload = self._as_result_data({(0, 0): raw}, 1, 1)
            payload["attrs"]["base_name"] = str(self.base_name)
            results[self._result_key] = (
                self._state_intervals(mdata, payload)
                if self.aggregate_states
                else _point_spatial_statistics(payload)
            )
        else:
            results[self._result_key] = raw
        return results

    def _join_target_blocks(
        self, blocks: list[Any]
    ) -> _Results | dict[str, list[_Results]]:
        """Assemble a complete point slab without losing interval statistics."""
        if self.aggregate_states:
            return dict(
                intervals=[
                    _combine_chunks(
                        [block["intervals"][i] for block in blocks], FC.TARGET
                    )
                    for i in range(len(blocks[0]["intervals"]))
                ]
            )
        if self.aggregate_points:
            return _merge_statistics(blocks)
        payload = self._as_result_data(
            {(0, target): block for target, block in enumerate(blocks)},
            1,
            len(blocks),
        )
        payload["attrs"]["base_name"] = str(self.base_name)
        return payload

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Join complete target slabs before applying the shared interval policy.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Total engine state chunks.
        n_chunks_targets
            Total engine target chunks per state slab.
        new_chunk_results
            Pending engine chunks under this writer's private key.
        output_results
            Run-local target tiles, interval buffers and write futures.

        Raises
        ------
        ValueError
            If a target chunk is duplicated or its indices are invalid.
        """
        tiles = output_results.setdefault(self._tiles_key, {})
        state = output_results.get(self._result_key, {}).get("next_chunk", 0)
        for key, payload in new_chunk_results.pop(self._result_key, {}).items():
            if (
                key in tiles
                or not state <= key[0] < n_chunks_states
                or not 0 <= key[1] < n_chunks_targets
            ):
                raise ValueError(f"{self.name}: Unexpected point chunk {key}")
            tiles[key] = payload

        # File intervals span complete point slabs, never a partial target tile.
        slabs: Dict[tuple[int, int], Any] = Dict()
        while state < n_chunks_states and all(
            (state, target) in tiles for target in range(n_chunks_targets)
        ):
            blocks = [tiles.pop((state, target)) for target in range(n_chunks_targets)]
            slabs[(state, 0)] = self._join_target_blocks(blocks)
            state += 1
        super().update_results(
            algo,
            n_chunks_states,
            n_chunks_targets,
            Dict({self._result_key: slabs}),
            output_results,
        )

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Reject incomplete target slabs, then await writes and remove buffers.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Expected engine state chunks.
        n_chunks_targets
            Expected engine target chunks.
        output_results
            Shared results, retaining only unrelated outputs after success.

        Raises
        ------
        ValueError
            If any target slab or state chunk is incomplete.
        """
        if output_results.pop(self._tiles_key):
            raise ValueError(f"{self.name}: Missing point result chunks")
        super().finalize_results(
            algo, n_chunks_states, n_chunks_targets, output_results
        )

    def _submit_file(
        self, engine: Engine, algo: Algorithm, fpath: Path, payload: _Results
    ) -> Any:
        payload = self._file_payload(payload)
        return engine.submit(
            _point_data_to_nc,
            fpath,
            payload,
            {"verbosity": algo.verbosity, **self.write_nc_pars},
        )


class PointResultsChunkWriter(PointResultsOutput):
    """One file per engine-supplied state/target chunk, written in its worker.

    Filenames end in ``_<state chunk:06d>_<target chunk:06d>.nc``. Fields and
    geometry retain every target-point position without rechunking. Requested
    aggregation removes the corresponding state/target dimensions of this tile.
    """

    def __init__(
        self,
        base_name: str | Path,
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure engine-chunk point files.

        Parameters
        ----------
        base_name
            Filename prefix followed by both six-digit engine chunk indices.
        write_nc_pars
            Options for ``foxes.utils.write_nc``, inheriting algorithm verbosity.
        kwargs
            Field selection, ``aggregate_points``, ``aggregate_states`` and
            path options for ``PointResultsOutput``.
        """
        self.base_name = base_name
        self.write_nc_pars = {} if write_nc_pars is None else dict(write_nc_pars)
        super().__init__(**kwargs)

    def calculate(  # type: ignore[override]
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData | None,
        model_results: Any,
        chunk_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Write selected fields from the supplied chunk and preserve shared results.

        Parameters
        ----------
        algo
            Executing algorithm, supplying inherited write verbosity.
        mdata
            Engine-supplied state and target chunk indices and totals.
        fdata
            Farm data, unused by this writer.
        tdata
            Required engine-supplied target chunk, read without mutation.
        model_results
            Calculated point fields.
        chunk_results
            Shared output mapping or ``None``.

        Returns
        -------
        chunk_results
            Shared mapping without any additional in-memory point dataset.

        Raises
        ------
        ValueError
            If target data or either engine chunk index is missing.

        Notes
        -----
        Worker write exceptions propagate directly.
        """
        if tdata is None:
            raise ValueError(f"{self.name}: Point output requires target data")
        if mdata.chunki_states is None or mdata.chunki_points is None:
            raise ValueError(f"{self.name}: Missing state or target chunk index")
        payload = self._as_result_data(
            {(0, 0): self._collect_chunk(mdata, tdata, model_results)}, 1, 1
        )
        payload = self._file_payload(payload)
        payload["attrs"].update(
            base_name=str(self.base_name),
            chunki_states=mdata.chunki_states,
            chunki_points=mdata.chunki_points,
            n_chunks_states=mdata.n_chunks_states,
            n_chunks_points=mdata.n_chunks_points,
        )
        path = self.get_fpath(
            f"{self.base_name}_{mdata.chunki_states:06d}_{mdata.chunki_points:06d}.nc"
        )
        _point_data_to_nc(
            path, payload, {"verbosity": algo.verbosity, **self.write_nc_pars}
        )
        return Output.calculate(
            self, algo, mdata, fdata, tdata, model_results, chunk_results
        )

    update_results = Output.update_results
    finalize_results = Output.finalize_results


class PointResultsPeriodWriter(_ResultsPeriodWriter, PointResultsCountWriter):
    """Point files grouped by calendar day, ISO week, month or year.

    Each file contains every target tile for its states, preserving all target
    points and state-dependent geometry. Chronology and filenames follow the
    same calendar policy as farm-result writers.
    """
