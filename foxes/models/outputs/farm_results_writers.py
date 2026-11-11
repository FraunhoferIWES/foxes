"""Chunk-local and calendar-period NetCDF farm writer policies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import foxes.constants as FC
from foxes.core import Output
from foxes.utils import Dict

from ._farm_result_data import (
    _FarmResults,
    _data_to_nc,
)

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


from .farm_results import FarmResultsCountWriter, FarmResultsOutput


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
        kwargs
            Farm selection, metadata and path options for ``FarmResultsOutput``.
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
        payload = self._collect_chunk(mdata, fdata)
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
        _data_to_nc(fpath, payload, *self._metadata(algo), self.write_nc_pars)
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


class FarmResultsPeriodWriter(FarmResultsCountWriter):
    """Farm files grouped by calendar day, ISO week, month or year."""

    def __init__(
        self,
        base_name: str | Path,
        *,
        period: Literal["day", "week", "month", "year"],
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure calendar-period farm files.

        Parameters
        ----------
        base_name
            Filename prefix followed by the calendar date or period.
        period
            Calendar grouping. Weeks begin on Monday and use the ISO week year.
            State coordinates must be nondecreasing, non-NaT ``numpy.datetime64``
            values. Empty periods do not produce files.
        write_nc_pars
            Keyword arguments forwarded to ``foxes.utils.write_nc``.
        kwargs
            Farm selection, metadata and path options for ``FarmResultsOutput``.

        Raises
        ------
        ValueError
            If ``period`` is not day, week, month or year.
        """
        # Validate the calendar selector before initializing the shared writer.
        if period not in {"day", "week", "month", "year"}:
            raise ValueError(
                f"Invalid period '{period}'; expected day, week, month or year"
            )
        super().__init__(
            base_name, n_states_per_file=1, write_nc_pars=write_nc_pars, **kwargs
        )
        self.period = period

    def _collect_chunk(self, mdata: MData, fdata: FData) -> _FarmResults:
        """Collect farm fields and attach the configured calendar-period name.

        Parameters
        ----------
        mdata
            Chunk metadata with explicit state coordinates.
        fdata
            Farm fields in original turbine order.

        Returns
        -------
        payload
            Independent farm payload whose attributes include ``period``.
        """
        payload = super()._collect_chunk(mdata, fdata)
        payload["attrs"]["period"] = self.period
        return payload

    def _validate_chunk(self, chunk: _FarmResults, previous: np.generic | None) -> None:
        """Require finite datetime coordinates in nondecreasing chunk order.

        Parameters
        ----------
        chunk
            Farm payload whose state coordinates are validated without mutation.
        previous
            Last state of the preceding chunk, or ``None`` for the first chunk.

        Raises
        ------
        ValueError
            If states are not datetime values, contain NaT, or decrease within
            or across chunks.
        """
        states = chunk["coords"][FC.STATE]
        if not np.issubdtype(states.dtype, np.datetime64):
            raise ValueError(
                f"{self.name}: State coordinates must be datetime64 values"
            )
        if np.any(np.isnat(states)):
            raise ValueError(f"{self.name}: State coordinates must not contain NaT")
        if np.any(states[1:] < states[:-1]) or (
            previous is not None and states[0] < previous
        ):
            raise ValueError(f"{self.name}: State coordinates must be chronological")

    def _next_period(self, first_time: np.datetime64) -> np.datetime64:
        """Return the exclusive calendar boundary after the first sample.

        Parameters
        ----------
        first_time
            Initial timestamp of the buffered interval.

        Returns
        -------
        boundary
            Start of the next day, ISO week, month or year. Week boundaries
            always fall on Monday, independent of the first sample's weekday.
        """
        # Align week boundaries with Monday rather than the first sample's weekday.
        day = first_time.astype("datetime64[D]")
        if self.period == "week":
            weekday = (int(day.astype(np.int64)) + 3) % 7
            return day + np.timedelta64(7 - weekday, "D")

        # Calendar units naturally align day, month and year boundaries.
        unit = {"day": "D", "month": "M", "year": "Y"}[self.period]
        return first_time.astype(f"datetime64[{unit}]") + np.timedelta64(1, unit)

    def _ready(self, chunks: list[_FarmResults]) -> bool:
        """Check whether buffered states have crossed a calendar boundary.

        Parameters
        ----------
        chunks
            Nonempty, chronological farm payloads.

        Returns
        -------
        ready
            Whether the final sample belongs to a later period than the first.
        """
        first = chunks[0]["coords"][FC.STATE][0]
        latest = chunks[-1]["coords"][FC.STATE][-1]
        return bool(latest >= self._next_period(first))

    def _file_size(self, states: np.ndarray) -> int:
        """Count buffered samples belonging to the first calendar period.

        Parameters
        ----------
        states
            Nonempty, nondecreasing datetime coordinates.

        Returns
        -------
        size
            Exclusive position of the next calendar boundary in ``states``.
        """
        return int(np.searchsorted(states, self._next_period(states[0]), side="left"))

    def _file_attrs(self, payload: _FarmResults) -> None:
        """Attach calendar and ISO date fields to one selected file interval.

        Parameters
        ----------
        payload
            Farm payload whose attributes are updated in place. Its first state
            determines calendar year/month/day and ISO week/year independently.
        """
        date = payload["coords"][FC.STATE][0].astype("datetime64[D]").astype(object)
        iso = date.isocalendar()
        payload["attrs"].update(
            year=date.year,
            month=date.month,
            day=date.day,
            week=iso.week,
            iso_year=iso.year,
        )

    def _get_fpath(self, attrs: dict[str, Any]) -> Path:
        """Resolve the filename for a calendar interval, using ISO week years.

        Parameters
        ----------
        attrs
            File attributes containing the calendar and ISO fields.

        Returns
        -------
        path
            Destination with a day, week, month or year suffix and ``.nc``.
        """
        if self.period == "day":
            suffix = f"{attrs['year']}-{attrs['month']:02d}-{attrs['day']:02d}"
        elif self.period == "week":
            suffix = f"{attrs['iso_year']}-W{attrs['week']:02d}"
        elif self.period == "month":
            suffix = f"{attrs['year']}-{attrs['month']:02d}"
        else:
            suffix = str(attrs["year"])
        return self.get_fpath(f"{self.base_name}_{suffix}.nc")
