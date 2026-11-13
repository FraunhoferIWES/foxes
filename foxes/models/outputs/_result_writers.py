"""Shared count-limited and calendar-period result writer policies."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

import foxes.constants as FC
from foxes.core import Output, get_engine
from foxes.utils import Dict

from ._result_data import _Results, _combine_chunks, _select_states
from ._aggregation import _merge_statistics, _state_statistics

if TYPE_CHECKING:
    from xarray import Dataset

    from foxes.core import Algorithm, Engine, MData


class _ResultsCountWriter(Output):
    """Buffered state intervals shared by farm and point result writers."""

    _RESULTS: str

    def __init__(
        self,
        base_name: str | Path,
        n_states_per_file: int,
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure count limits, file options and the concrete collector.

        Parameters
        ----------
        base_name
            File prefix; filenames append ``_<file index:06d>.nc``.
        n_states_per_file
            Positive integer state limit, except for the final remainder.
        write_nc_pars
            Options forwarded to ``foxes.utils.write_nc``; verbosity inherits
            the executing algorithm unless explicitly overridden.
        kwargs
            Selection, metadata and path options for the concrete collector.

        Raises
        ------
        ValueError
            If the state limit is not a positive integer.
        """
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
        self._result_key = self.var(self._RESULTS)

    def get_initial_output_results(
        self,
        algo: Algorithm,
        model_data: Dataset | None = None,
        farm_data: Dataset | None = None,
        point_data: Dataset | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Store parent-only NetCDF options before worker execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        model_data
            Initial model dataset passed to the base output hook.
        farm_data
            Initial farm dataset passed to the base output hook.
        point_data
            Initial target dataset, or ``None`` for farm calculations.
        extra_data
            Additional non-array inputs passed to the base output hook.

        Returns
        -------
        output_results
            Run-local results containing this writer's private NetCDF option
            key, referencing its original configuration object.
        """
        output_results = super().get_initial_output_results(
            algo, model_data, farm_data, point_data, extra_data
        )

        # Bypass recursive Dict conversion to retain exact option object identity.
        dict.update(output_results, {self.var("write_nc_pars"): self.write_nc_pars})
        return output_results

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Move NetCDF options off the model while it is running.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage, or ``None`` to retain model state.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().set_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            data_stash[self.name]["write_nc_pars"] = self.write_nc_pars
            del self.write_nc_pars

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """Restore NetCDF options after execution.

        Parameters
        ----------
        algo
            The calculation algorithm.
        data_stash
            Model-name keyed lifecycle storage containing this writer's options.
        sel
            Selected states for this calculation, if any.
        verbosity
            The lifecycle reporting level; ``0`` is silent.
        """
        super().unset_running(algo, data_stash, sel, verbosity)
        if data_stash is not None:
            self.write_nc_pars = data_stash[self.name].pop("write_nc_pars")

    def _ready(self, chunks: list[_Results]) -> bool:
        """Check whether buffered samples fill the configured state limit."""
        return (
            sum(len(chunk["coords"][FC.STATE]) for chunk in chunks)
            >= self.n_states_per_file
        )

    def _file_size(self, states: np.ndarray) -> int:
        """Select no more than the configured number of states."""
        return min(self.n_states_per_file, len(states))

    def _get_fpath(self, attrs: dict[str, Any]) -> Path:
        """Resolve the next count-limited filename."""
        return self.get_fpath(f"{self.base_name}_{attrs['index']:06d}.nc")

    def _file_attrs(self, payload: _Results) -> None:
        """Attach the configured count limit to one file interval."""
        payload["attrs"]["n_states_per_file"] = self.n_states_per_file

    def _validate_chunk(self, chunk: _Results, previous: np.generic | None) -> None:
        """Accept state coordinates without additional count-policy restrictions."""

    def _interval_size(self, states: np.ndarray, offset: int) -> int:
        """Partition workers at global count boundaries, not engine boundaries."""
        return min(
            self.n_states_per_file - offset % self.n_states_per_file, len(states)
        )

    def _interval_key(self, first: np.generic, offset: int) -> int | str:
        """Identify a count interval independently of chunk arrival."""
        return offset // self.n_states_per_file

    def _state_intervals(self, mdata: MData, payload: _Results) -> dict[str, Any]:
        """Reduce each worker-local file interval without returning state histories."""
        states = payload["coords"][FC.STATE]
        if not len(states):
            raise ValueError(f"{self.name}: Cannot write an empty state chunk")
        self._validate_chunk(payload, None)
        offset = mdata.states_i0(counter=True)
        if offset is None:
            raise ValueError(
                f"{self.name}: State interval reduction requires a chunk offset"
            )
        intervals = []
        start = 0
        while start < len(states):
            size = self._interval_size(states[start:], offset + start)
            selected = _select_states(payload, start, start + size)
            self._file_attrs(selected)
            stats = _state_statistics(selected)
            stats["attrs"].update(
                __interval_key=self._interval_key(states[start], offset + start),
                __interval_first=states[start],
                __interval_last=states[start + size - 1],
            )
            intervals.append(stats)
            start += size
        return dict(intervals=intervals)

    def _flush_statistics(self, algo: Algorithm, results: dict[str, Any]) -> None:
        """Write one complete interval after merging per-entity sufficient statistics."""
        engine = get_engine(error=False)
        if engine is None:
            raise RuntimeError(f"{self.name}: File writing requires an active engine")
        payload = _merge_statistics(results["chunks"])
        payload["attrs"] = {
            name: value
            for name, value in payload["attrs"].items()
            if not name.startswith("__interval_")
        }
        payload["attrs"]["index"] = results["index"]
        results["futures"].append(
            self._submit_file(
                engine,
                algo,
                self._get_fpath(payload["attrs"]),
                payload,
                results["write_nc_pars"],
            )
        )
        results["index"] += 1
        results["chunks"] = []

    def _append_intervals(
        self, algo: Algorithm, results: dict[str, Any], chunk: dict[str, Any]
    ) -> None:
        """Keep only the unfinished file's statistics, never its individual samples."""
        for interval in chunk["intervals"]:
            attrs = interval["attrs"]
            bounds: _Results = dict(
                coords={
                    FC.STATE: np.asarray(
                        [attrs["__interval_first"], attrs["__interval_last"]]
                    )
                },
                data_vars={},
                attrs={},
            )
            self._validate_chunk(bounds, results["last_state"])
            results["last_state"] = attrs["__interval_last"]
            if (
                results["chunks"]
                and results["chunks"][0]["attrs"]["__interval_key"]
                != attrs["__interval_key"]
            ):
                self._flush_statistics(algo, results)
            results["chunks"].append(interval)
            if len(results["chunks"]) > 1:
                results["chunks"] = [_merge_statistics(results["chunks"])]
        if results["last"] and results["chunks"]:
            self._flush_statistics(algo, results)

    @abstractmethod
    def _submit_file(
        self,
        engine: Engine,
        algo: Algorithm,
        fpath: Path,
        payload: _Results,
        write_nc_pars: dict[str, Any],
    ) -> Any:
        """Submit one independent file using the concrete dataset serializer.

        Parameters
        ----------
        engine
            Active execution engine receiving the file job.
        algo
            Executing algorithm supplying metadata and inherited verbosity.
        fpath
            Fully resolved destination path.
        payload
            Selected NumPy fields and copied file attributes.
        write_nc_pars
            Parent-owned NetCDF options.

        Returns
        -------
        future
            Engine-owned handle whose write failure must propagate at finalization.
        """
        raise NotImplementedError

    def _flush(self, algo: Algorithm, results: dict[str, Any]) -> None:
        """Submit complete intervals and retain only their unfinished remainder."""
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

            # Keep each submitted file independent of the retained run buffer.
            selected = _select_states(payload, start, start + size)
            selected["attrs"].update(index=results["index"], n_states=size)
            self._file_attrs(selected)
            results["futures"].append(
                self._submit_file(
                    engine,
                    algo,
                    self._get_fpath(selected["attrs"]),
                    selected,
                    results["write_nc_pars"],
                )
            )
            results["index"] += 1
            start += size
        results["chunks"] = (
            [_select_states(payload, start, n_states)] if start < n_states else []
        )

    def update_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        new_chunk_results: Dict[str, Dict[tuple[int, int], Any]],
        output_results: Dict[str, Any],
    ) -> None:
        """Buffer complete state slabs and submit every ready file interval.

        Parameters
        ----------
        algo
            Executing algorithm and metadata.
        n_chunks_states
            Total state chunks; the last flushes the remaining samples.
        n_chunks_targets
            Total target chunks, already unified by point writers.
        new_chunk_results
            Pending slabs under this writer's private result key.
        output_results
            Run-local buffers, futures and interval indices, updated in place.

        Raises
        ------
        ValueError
            If a state chunk is empty or arrives out of order.
        """
        chunks = new_chunk_results.pop(self._result_key, Dict())
        results = output_results.setdefault(
            self._result_key,
            dict(
                chunks=[],
                index=0,
                next_chunk=0,
                futures=[],
                last=False,
                last_state=None,
                write_nc_pars=output_results[self.var("write_nc_pars")],
            ),
        )
        for key, chunk in chunks.items():
            expected = (results["next_chunk"], 0)
            if key != expected:
                raise ValueError(
                    f"{self.name}: Unexpected chunk {key}, expected {expected}"
                )
            if "intervals" in chunk:
                results["next_chunk"] += 1
                results["last"] = results["next_chunk"] == n_chunks_states
                self._append_intervals(algo, results, chunk)
                continue
            states = chunk["coords"][FC.STATE]
            if not len(states):
                raise ValueError(f"{self.name}: Cannot write an empty state chunk")
            self._validate_chunk(chunk, results["last_state"])
            results["last_state"] = states[-1]
            results["chunks"].append(chunk)
            results["next_chunk"] += 1
            results["last"] = results["next_chunk"] == n_chunks_states
            if self._ready(results["chunks"]) or results["last"]:
                self._flush(algo, results)

    def finalize_results(
        self,
        algo: Algorithm,
        n_chunks_states: int,
        n_chunks_targets: int,
        output_results: Dict[str, Any],
    ) -> None:
        """Await all file jobs and remove the private writer buffer.

        Parameters
        ----------
        algo
            Executing algorithm.
        n_chunks_states
            Expected number of complete state slabs.
        n_chunks_targets
            Total target chunks.
        output_results
            Shared results; unrelated outputs are left unchanged.

        Raises
        ------
        ValueError
            If not all expected state slabs were received.
        RuntimeError
            If no engine is active to await writes.

        Notes
        -----
        Worker write exceptions propagate when submitted futures are awaited.
        """
        output_results.pop(self.var("write_nc_pars"))
        results = output_results.pop(self._result_key, None)
        if results is None:
            if n_chunks_states:
                raise ValueError(f"{self.name}: Missing result chunks")
            return
        if results["next_chunk"] != n_chunks_states:
            raise ValueError(f"{self.name}: Missing result chunks")
        engine = get_engine(error=False)
        if engine is None:
            raise RuntimeError(f"{self.name}: File writing requires an active engine")
        engine.await_results(results["futures"])


class _ResultsPeriodWriter(_ResultsCountWriter):
    """Calendar boundaries shared by buffered farm and point writers."""

    def __init__(
        self,
        base_name: str | Path,
        *,
        period: Literal["day", "week", "month", "year"],
        write_nc_pars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure chronological calendar-period files.

        Parameters
        ----------
        base_name
            Filename prefix followed by a calendar date or period.
        period
            Day, Monday-to-Sunday ISO week, month or year. States must be
            nondecreasing non-NaT ``numpy.datetime64`` values. Empty periods
            produce no files.
        write_nc_pars
            Options for ``foxes.utils.write_nc``, inheriting algorithm verbosity.
        kwargs
            Selection, metadata and path options for the concrete collector.

        Raises
        ------
        ValueError
            If the period is unsupported.
        """
        if period not in {"day", "week", "month", "year"}:
            raise ValueError(
                f"Invalid period '{period}'; expected day, week, month or year"
            )
        super().__init__(
            base_name, n_states_per_file=1, write_nc_pars=write_nc_pars, **kwargs
        )
        self.period = period

    def _validate_chunk(self, chunk: _Results, previous: np.generic | None) -> None:
        """Require finite datetime coordinates in nondecreasing state order."""
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
        """Return the exclusive calendar boundary following the first sample."""
        day = first_time.astype("datetime64[D]")
        if self.period == "week":
            weekday = (int(day.astype(np.int64)) + 3) % 7
            return day + np.timedelta64(7 - weekday, "D")
        unit = {"day": "D", "month": "M", "year": "Y"}[self.period]
        return first_time.astype(f"datetime64[{unit}]") + np.timedelta64(1, unit)

    def _interval_size(self, states: np.ndarray, offset: int) -> int:
        """Partition worker samples at calendar boundaries."""
        return self._file_size(states)

    def _interval_key(self, first: np.generic, offset: int) -> str:
        """Identify a calendar interval by its exclusive upper boundary."""
        return str(self._next_period(first))

    def _ready(self, chunks: list[_Results]) -> bool:
        """Check whether buffered states have crossed a calendar boundary."""
        first = chunks[0]["coords"][FC.STATE][0]
        latest = chunks[-1]["coords"][FC.STATE][-1]
        return bool(latest >= self._next_period(first))

    def _file_size(self, states: np.ndarray) -> int:
        """Count buffered samples belonging to the first calendar period."""
        return int(np.searchsorted(states, self._next_period(states[0]), side="left"))

    def _file_attrs(self, payload: _Results) -> None:
        """Attach calendar and ISO fields determined by the interval's first state."""
        date = payload["coords"][FC.STATE][0].astype("datetime64[D]").astype(object)
        iso = date.isocalendar()
        payload["attrs"].update(
            period=self.period,
            year=date.year,
            month=date.month,
            day=date.day,
            week=iso.week,
            iso_year=iso.year,
        )

    def _get_fpath(self, attrs: dict[str, Any]) -> Path:
        """Resolve a calendar filename, using the ISO year for week suffixes."""
        if self.period == "day":
            suffix = f"{attrs['year']}-{attrs['month']:02d}-{attrs['day']:02d}"
        elif self.period == "week":
            suffix = f"{attrs['iso_year']}-W{attrs['week']:02d}"
        elif self.period == "month":
            suffix = f"{attrs['year']}-{attrs['month']:02d}"
        else:
            suffix = str(attrs["year"])
        return self.get_fpath(f"{self.base_name}_{suffix}.nc")
