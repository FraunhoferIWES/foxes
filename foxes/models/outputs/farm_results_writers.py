"""Chunk-local and calendar-period NetCDF farm writer policies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import foxes.constants as FC
from foxes.core import Output
from foxes.utils import Dict

from ._result_data import _Results, _data_to_nc
from ._result_writers import _ResultsCountWriter, _ResultsPeriodWriter

if TYPE_CHECKING:
    from foxes.core import Algorithm, Engine
    from foxes.core.data import FData, MData, TData


from .farm_results import FarmResultsOutput


class FarmResultsCountWriter(_ResultsCountWriter, FarmResultsOutput):
    """Farm files with a fixed state limit, independent of engine chunk sizes."""

    _RESULTS = FC.FARM_RESULTS

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
            Field selection, ``aggregate_turbines``, ``aggregate_states``,
            metadata and path options for ``FarmResultsOutput``.

        Raises
        ------
        ValueError
            If ``n_states_per_file`` is not a positive integer.
        """
        super().__init__(
            base_name, n_states_per_file, write_nc_pars=write_nc_pars, **kwargs
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
        payload = self._collect_chunk(algo, mdata, fdata, reduce_states=False)
        payload["attrs"].update(base_name=str(self.base_name))
        chunk_results[self._result_key] = (
            self._state_intervals(mdata, payload) if self.aggregate_states else payload
        )
        return chunk_results

    def _submit_file(
        self,
        engine: Engine,
        algo: Algorithm,
        fpath: Path,
        payload: _Results,
        write_nc_pars: dict[str, Any],
    ) -> Any:
        if self.aggregate_states:
            payload = self._file_payload(algo, payload)
        return engine.submit(
            _data_to_nc,
            fpath,
            payload,
            *self._metadata(algo),
            {"verbosity": algo.verbosity, **write_nc_pars},
        )


class FarmResultsChunkWriter(FarmResultsOutput):
    """One NetCDF file per engine state chunk, without a returned dataset."""

    def __init__(
        self,
        base_name: str | Path,
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure per-chunk farm files.

        Parameters
        ----------
        base_name
            File prefix; filenames append ``_<state chunk index:06d>.nc``.
        write_nc_pars
            Keyword arguments forwarded to ``foxes.utils.write_nc``.
            Verbosity defaults to the executing algorithm's level.
        kwargs
            Field selection, ``aggregate_turbines``, ``aggregate_states``,
            metadata and path options for ``FarmResultsOutput``.
        """
        # Retain independent file options for this writer.
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
        """Write this chunk's selected farm arrays and preserve shared results.

        Parameters
        ----------
        algo
            The calculation algorithm and farm metadata.
        mdata
            Chunk indices, counts and state coordinates.
        fdata
            Farm-ordered arrays to select and write.
        tdata
            Unused target data; may be ``None``.
        model_results
            Unused model return values.
        chunk_results
            Shared results, returned unchanged or created when ``None``.

        Returns
        -------
        chunk_results
            The shared mapping, without an additional in-memory farm dataset.

        Notes
        -----
        Write failures propagate from the worker. Files contain the base name,
        chunk indices and total chunk counts as attributes.
        """
        # Build this writer's selection independently of in-memory outputs.
        if mdata.chunki_points != 0:
            return Output.calculate(
                self, algo, mdata, fdata, tdata, model_results, chunk_results
            )
        payload = self._collect_chunk(algo, mdata, fdata)
        if self.aggregate_states:
            payload = self._file_payload(algo, payload)
        payload["attrs"].update(
            base_name=str(self.base_name),
            chunki_states=mdata.chunki_states,
            chunki_points=mdata.chunki_points,
            n_chunks_states=mdata.n_chunks_states,
            n_chunks_points=mdata.n_chunks_points,
        )

        # Write in the chunk worker before returning other outputs' payloads.
        if mdata.chunki_states is None:
            raise ValueError(f"{self.name}: Missing state chunk index")
        fpath = self.get_fpath(f"{self.base_name}_{mdata.chunki_states:06d}.nc")
        options = {"verbosity": algo.verbosity, **self.write_nc_pars}
        _data_to_nc(fpath, payload, *self._metadata(algo), options)
        return Output.calculate(
            self, algo, mdata, fdata, tdata, model_results, chunk_results
        )

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Leave shared buffers unchanged after worker-local files are written.

        Parameters
        ----------
        algo
            Executing algorithm; unused by this no-op hook.
        n_chunks_states
            Total state chunks; unused.
        n_chunks_targets
            Total target chunks; unused.
        new_chunk_results
            Shared pending results, not consumed by this writer.
        output_results
            Run-local results, left unchanged because files were written during
            ``calculate`` rather than buffered for parent-side assembly.
        """
        return Output.update_results(
            self,
            algo,
            n_chunks_states,
            n_chunks_targets,
            new_chunk_results,
            output_results,
        )

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Leave public results unchanged after synchronous worker-local writes.

        Parameters
        ----------
        algo
            Executing algorithm; unused by this no-op hook.
        n_chunks_states
            Total state chunks; unused.
        n_chunks_targets
            Total target chunks; unused.
        output_results
            Shared results, left unchanged. This writer has no deferred futures
            or in-memory farm dataset to finalize.
        """
        return Output.finalize_results(
            self, algo, n_chunks_states, n_chunks_targets, output_results
        )


class FarmResultsPeriodWriter(_ResultsPeriodWriter, FarmResultsCountWriter):
    """Farm files grouped by calendar day, ISO week, month or year."""
