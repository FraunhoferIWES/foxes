"""Chunk-local farm result collection and NetCDF output models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from xarray import Dataset

from foxes.config import config
from foxes.core import Output, get_engine
from foxes.utils import Dict, write_nc
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class _FarmResults(TypedDict):
    coords: dict[str, np.ndarray]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
    attrs: dict[str, Any]


def _combine_chunks(chunks: list[_FarmResults]) -> _FarmResults:
    """Join ordered farm payloads without mutating the source chunks."""
    # Validate variable schemas before concatenating state arrays.
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk["data_vars"].keys() != first["data_vars"].keys():
            raise ValueError("Farm results: Variable selections differ between chunks")
        for variable, (dims, _) in chunk["data_vars"].items():
            if dims != first["data_vars"][variable][0]:
                raise ValueError(
                    f"Farm results: Dimensions differ between chunks for '{variable}'"
                )

    # Preserve static coordinates and concatenate along the state axis.
    coords = first["coords"].copy()
    coords[FC.STATE] = np.concatenate([chunk["coords"][FC.STATE] for chunk in chunks])
    data_vars = {
        variable: (
            dims,
            np.concatenate(
                [chunk["data_vars"][variable][1] for chunk in chunks], axis=0
            ),
        )
        for variable, (dims, _) in first["data_vars"].items()
    }
    return dict(coords=coords, data_vars=data_vars, attrs=first["attrs"].copy())


def _select_states(payload: _FarmResults, start: int, stop: int) -> _FarmResults:
    """Select a state interval while retaining static turbine coordinates."""
    # Slice each state-dependent array without modifying the full payload.
    return dict(
        coords={
            name: values[start:stop] if name == FC.STATE else values
            for name, values in payload["coords"].items()
        },
        data_vars={
            name: (dims, values[start:stop])
            for name, (dims, values) in payload["data_vars"].items()
        },
        attrs=payload["attrs"].copy(),
    )


def _create_dataset(
    payload: _FarmResults,
    turbine_names: Sequence[str] | None,
    wind_farm_list: Sequence[str | None] | None,
    cluster_list: Sequence[str | None] | None,
) -> Dataset:
    """Create a dataset with farm metadata and compact uniform state weights."""
    # Create the dataset and attach only requested, populated metadata.
    dataset = Dataset(
        data_vars=payload["data_vars"],
        coords=payload["coords"],
        attrs=payload["attrs"],
    )
    for name, values in [
        (FC.TNAME, turbine_names),
        (FC.FARM, wind_farm_list),
        (FC.CLUSTER, cluster_list),
    ]:
        if values is not None and any(value is not None for value in values):
            dataset[name] = ((FC.TURBINE,), values)

    # Compress weights only after all states for this dataset are available.
    if FV.WEIGHT in dataset:
        weights = dataset[FV.WEIGHT].to_numpy()
        if weights.shape[1] > 0 and np.all(weights == weights[:, :1]):
            dataset[FV.WEIGHT] = ((FC.STATE,), weights[:, 0])
    if FV.ORDER in dataset:
        dataset[FV.ORDER] = dataset[FV.ORDER].astype(config.dtype_int)
    return dataset


def _data_to_nc(
    fpath: Path,
    payload: _FarmResults,
    turbine_names: Sequence[str] | None,
    wind_farm_list: Sequence[str | None] | None,
    cluster_list: Sequence[str | None] | None,
    write_nc_pars: dict[str, Any],
) -> None:
    """Create and write one farm dataset in the engine's worker context."""
    # Use the same dataset construction for all writer variants.
    dataset = _create_dataset(payload, turbine_names, wind_farm_list, cluster_list)
    write_nc(ds=dataset, fpath=fpath, **write_nc_pars)


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
        """Collect farm-ordered arrays and preserve explicit state coordinates."""
        # Retain dimensions even when all physical fields are filtered out.
        coords = {
            FC.STATE: mdata[FC.STATE] if FC.STATE in mdata else fdata[FC.STATE],
            FC.TURBINE: fdata[FC.TURBINE],
        }
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        variables = (
            list(fdata) if self.fdata_variables is None else self.fdata_variables
        )
        excluded = {
            FC.STATE,
            FC.TURBINE,
            FC.FARM,
            FC.CLUSTER,
            FC.XYH,
            FV.TXYH,
            FC.TNAME,
        }

        # Filter physical variables without changing chunk-local weight dimensions.
        for variable in variables:
            if variable in excluded:
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
        """Select farm metadata consistently for datasets and files."""
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
        if FC.FARM_RESULTS not in chunk_results:
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
        chunks = new_chunk_results.pop(FC.FARM_RESULTS)
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
            Must be one for farm output.
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
        if n_chunks_targets != 1:
            raise ValueError(
                f"{self.name}: Expected one target chunk, got {n_chunks_targets}"
            )
        chunks = output_results[FC.FARM_RESULTS]
        payload = _combine_chunks(
            [chunks[(index, 0)] for index in range(n_chunks_states)]
        )
        output_results[FC.FARM_RESULTS] = _create_dataset(
            payload, *self._metadata(algo)
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
        return Output.finalize_results(
            self, algo, n_chunks_states, n_chunks_targets, output_results
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
        """Submit complete files and retain only the incomplete state interval."""
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
                self.write_nc_pars,
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
            Must be one for farm output.
        new_chunk_results
            Shared pending payloads. Only this writer's private key is consumed.
        output_results
            Updated with private buffers and write futures, not a full dataset.

        Raises
        ------
        ValueError
            If target chunks are present or state chunks arrive out of order.
        """
        # Require ordered farm chunks and initialize run-local writer buffers.
        if n_chunks_targets != 1:
            raise ValueError(
                f"{self.name}: Expected one target chunk, got {n_chunks_targets}"
            )
        chunks = new_chunk_results.pop(self.FARM_RESULTS)
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
        """Validate writer-specific state requirements before buffering a chunk."""

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
        for future in results["futures"]:
            engine.await_result(future)


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
        # Attach the period to this writer's independently collected payload.
        payload = super()._collect_chunk(mdata, fdata)
        payload["attrs"]["period"] = self.period
        return payload

    def _validate_chunk(self, chunk: _FarmResults, previous: np.generic | None) -> None:
        # Require chronological datetime coordinates within and across chunks.
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
        """Return the exclusive calendar boundary after the first sample."""
        # Align week boundaries with Monday rather than the first sample's weekday.
        day = first_time.astype("datetime64[D]")
        if self.period == "week":
            weekday = (int(day.astype(np.int64)) + 3) % 7
            return day + np.timedelta64(7 - weekday, "D")

        # Calendar units naturally align day, month and year boundaries.
        unit = {"day": "D", "month": "M", "year": "Y"}[self.period]
        return first_time.astype(f"datetime64[{unit}]") + np.timedelta64(1, unit)

    def _ready(self, chunks: list[_FarmResults]) -> bool:
        # Write once the buffered samples enter a later calendar period.
        first = chunks[0]["coords"][FC.STATE][0]
        latest = chunks[-1]["coords"][FC.STATE][-1]
        return bool(latest >= self._next_period(first))

    def _file_size(self, states: np.ndarray) -> int:
        # Split each buffered interval at its own next calendar boundary.
        return int(np.searchsorted(states, self._next_period(states[0]), side="left"))

    def _file_attrs(self, payload: _FarmResults) -> None:
        # Record calendar and ISO fields without confusing their year conventions.
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
        # Name weekly files with the ISO year, including December/January crossings.
        if self.period == "day":
            suffix = f"{attrs['year']}-{attrs['month']:02d}-{attrs['day']:02d}"
        elif self.period == "week":
            suffix = f"{attrs['iso_year']}-W{attrs['week']:02d}"
        elif self.period == "month":
            suffix = f"{attrs['year']}-{attrs['month']:02d}"
        else:
            suffix = str(attrs["year"])
        return self.get_fpath(f"{self.base_name}_{suffix}.nc")
