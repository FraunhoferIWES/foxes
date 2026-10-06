from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import get_input_path
from foxes.data import STATES, StaticData
from foxes.models.vertical_profiles.abl_log_ws import ABLLogWsProfile

from .field_data import LatLonFieldData


_METADATA_ROWS = 8


@dataclass(frozen=True)
class _PECDFileData:
    point_ids: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    states: np.ndarray
    values: np.ndarray


def _read_decimal_row(row: list[str], label: str, path: Path) -> np.ndarray:
    try:
        values = np.asarray(
            [float(value.strip().replace(",", ".")) for value in row[1:]],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError(
            f"PECDStates: Invalid decimal values for {label} in '{path}'"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"PECDStates: Non-finite {label} values in '{path}'")
    return values


def _read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read PECD point IDs and geographic coordinates."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(islice(csv.reader(stream, delimiter=";"), _METADATA_ROWS))
    if len(rows) != _METADATA_ROWS:
        raise ValueError(f"PECDStates: Incomplete metadata in '{path}'")

    for row_index, label in (
        (4, "!MESSHOEHE_M"),
        (5, "!BREITENGRAD_DEG_N"),
        (6, "!LAENGENGRAD_DEG_E"),
        (7, "NNF_ID"),
    ):
        if not rows[row_index] or rows[row_index][0] != label:
            raise ValueError(f"PECDStates: Expected metadata row '{label}' in '{path}'")

    measurement_heights = _read_decimal_row(rows[4], "measurement height", path)
    latitudes = _read_decimal_row(rows[5], "latitudes", path)
    longitudes = _read_decimal_row(rows[6], "longitudes", path)
    try:
        point_ids = np.asarray([int(value) for value in rows[7][1:]], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"PECDStates: Invalid point IDs in '{path}'") from exc

    if not (
        len(point_ids) == len(measurement_heights) == len(latitudes) == len(longitudes)
    ):
        raise ValueError(f"PECDStates: Inconsistent point metadata in '{path}'")
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError(f"PECDStates: Duplicate point IDs in '{path}'")
    return point_ids, latitudes, longitudes


def _read_records(path: Path, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Read sequential state IDs and numerical values after PECD metadata."""
    try:
        table = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            header=None,
            skiprows=_METADATA_ROWS,
        )
        if table.shape[1] != n_points + 1:
            raise ValueError(
                f"Expected {n_points + 1} record columns, got {table.shape[1]}"
            )
        raw_states = pd.to_numeric(table.iloc[:, 0], errors="raise").to_numpy(
            dtype=np.float64
        )
        values = table.iloc[:, 1:].to_numpy(dtype=np.float64)
    except (pd.errors.ParserError, TypeError, ValueError) as exc:
        raise ValueError(f"PECDStates: Invalid records in '{path}': {exc}") from exc

    states = raw_states.astype(np.int64)
    if not np.array_equal(raw_states, states) or np.any(np.diff(states) <= 0):
        raise ValueError(f"PECDStates: Invalid or unsorted state IDs in '{path}'")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"PECDStates: Non-finite input values in '{path}'")
    return states, values


def _to_grid(
    file_data: _PECDFileData, latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Arrange point-column values on the complete regular PECD grid."""
    lat_indices = np.searchsorted(latitudes, file_data.latitudes)
    lon_indices = np.searchsorted(longitudes, file_data.longitudes)
    pairs = np.column_stack((lat_indices, lon_indices))
    if len(latitudes) * len(longitudes) != len(file_data.point_ids) or len(
        np.unique(pairs, axis=0)
    ) != len(file_data.point_ids):
        raise ValueError("PECDStates: Point coordinates do not form a complete grid")

    grid = np.empty(
        (len(file_data.states), len(latitudes), len(longitudes)), dtype=np.float64
    )
    grid[:, lat_indices, lon_indices] = file_data.values
    return grid


def _profile_wind_speed(
    wind_speed: np.ndarray,
    reference_height: float,
    extrapolation_heights: list[float] | None,
    profile_z0: float | None,
    profile_mol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build requested wind-speed heights once using the ABL log profile."""
    requested = np.asarray(
        [] if extrapolation_heights is None else extrapolation_heights,
        dtype=np.float64,
    )
    if (
        requested.ndim != 1
        or not np.all(np.isfinite(requested))
        or np.any(requested <= 0)
    ):
        raise ValueError(
            "PECDStates: extrapolation_heights must contain finite positive heights"
        )

    heights = np.unique(np.concatenate(([reference_height], requested)))
    extra_heights = heights[heights != reference_height]
    if len(extra_heights) == 0:
        return heights, wind_speed[None, ...]

    if profile_z0 is None or not np.isfinite(profile_z0) or profile_z0 <= 0:
        raise ValueError(
            "PECDStates: profile_z0 must be finite and positive when extrapolating"
        )
    if reference_height <= profile_z0 or np.any(extra_heights <= profile_z0):
        raise ValueError("PECDStates: profile heights must exceed profile_z0")
    if np.isinf(profile_mol):
        raise ValueError("PECDStates: profile_mol must be finite or NaN")

    profile = ABLLogWsProfile()
    profile_data = {
        FV.WS: wind_speed,
        FV.H: reference_height,
        FV.Z0: profile_z0,
        FV.MOL: profile_mol,
    }
    profiled = np.empty((len(heights), *wind_speed.shape), dtype=np.float64)
    for index, target_height in enumerate(heights):
        if target_height == reference_height:
            profiled[index] = wind_speed
        else:
            profile_heights = np.full_like(wind_speed, target_height)
            profiled[index] = profile.calculate(profile_data, profile_heights)
    return heights, profiled


def _resolve_input_file(filename: str | Path, variable: str) -> Path:
    """Resolve a PECD input path, then search packaged states by basename."""
    path = get_input_path(filename)
    if path.is_file():
        return path
    packaged_path = StaticData().get_file_path(
        STATES, path.name, check_raw=False, errors=False
    )
    if packaged_path is not None:
        return packaged_path
    raise FileNotFoundError(
        f"PECDStates: {variable} file not found: '{path}'; "
        f"no packaged states file named '{path.name}'"
    )


def _read_pecd_dataset(
    wind_speed_file: str | Path,
    wind_direction_file: str | Path,
    height: float,
    extrapolation_heights: list[float] | None,
    profile_z0: float | None,
    profile_mol: float,
) -> xr.Dataset:
    """Read explicit PECD wind files into the DatasetStates data contract."""
    if not np.isfinite(height) or height <= 0:
        raise ValueError("PECDStates: height must be finite and positive")

    files: dict[str, _PECDFileData] = {}
    reference_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    reference_states: np.ndarray | None = None
    input_files = {
        FV.WS: _resolve_input_file(wind_speed_file, FV.WS),
        FV.WD: _resolve_input_file(wind_direction_file, FV.WD),
    }
    for key, path in input_files.items():
        point_ids, point_latitudes, point_longitudes = _read_metadata(path)
        states, values = _read_records(path, len(point_ids))
        if reference_grid is None:
            reference_grid = (point_ids, point_latitudes, point_longitudes)
            reference_states = states
        elif (
            not np.array_equal(point_ids, reference_grid[0])
            or not np.array_equal(point_latitudes, reference_grid[1])
            or not np.array_equal(point_longitudes, reference_grid[2])
        ):
            raise ValueError(f"PECDStates: Grid metadata differs in '{path}'")
        elif not np.array_equal(states, reference_states):
            raise ValueError(f"PECDStates: State IDs differ in '{path}'")
        files[key] = _PECDFileData(
            point_ids, point_latitudes, point_longitudes, states, values
        )

    assert reference_grid is not None and reference_states is not None
    latitudes = np.unique(reference_grid[1])
    longitudes = np.unique(reference_grid[2])
    grids = {
        key: _to_grid(file_data, latitudes, longitudes)
        for key, file_data in files.items()
    }
    heights, wind_speed = _profile_wind_speed(
        grids[FV.WS],
        height,
        extrapolation_heights,
        profile_z0,
        profile_mol,
    )
    wind_direction = np.broadcast_to(
        grids[FV.WD][:, None, :, :],
        (len(reference_states), len(heights), len(latitudes), len(longitudes)),
    )
    return xr.Dataset(
        data_vars={
            FV.WS: (
                (FC.STATE, "height", "latitude", "longitude"),
                np.moveaxis(wind_speed, 0, 1),
            ),
            FV.WD: (
                (FC.STATE, "height", "latitude", "longitude"),
                wind_direction,
            ),
        },
        coords={
            FC.STATE: reference_states,
            "height": heights,
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


class PECDStates(LatLonFieldData):
    """PECD wind states from semicolon-delimited CSV files on a regular grid.

    Input paths are checked in FOXES' configured input directory, then missing
    files are looked up by basename in the packaged states data. Both wind
    fields are assigned the configured ``height`` coordinate, which defaults
    to 100 m and overrides the measurement-height metadata in each CSV. CSV
    record IDs become state labels.

    Optional ``extrapolation_heights`` are generated once while the CSV data is
    loaded, using ``ABLLogWsProfile``. The reference-height wind speed is kept
    unchanged; wind direction is treated as height-independent. Extrapolation
    requires ``profile_z0``. ``profile_mol`` defaults to NaN, selecting a
    neutral profile.

    Examples
    --------
    >>> states = PECDStates(
    ...     "pecd_wind_speed.csv",
    ...     "pecd_wind_direction.csv",
    ... )
    >>> states.data_source.sizes[FC.STATE]
    240
    """

    def __init__(
        self,
        wind_speed_file: str | Path,
        wind_direction_file: str | Path,
        height: float = 100.0,
        extrapolation_heights: list[float] | None = None,
        profile_z0: float | None = None,
        profile_mol: float = np.nan,
        output_vars: list[str] | None = None,
        load_mode: str = "preload",
        **kwargs: Any,
    ) -> None:
        """Initialize PECD wind states from the required CSV files.

        Parameters
        ----------
        wind_speed_file
            Path or packaged states filename for the wind-speed CSV.
        wind_direction_file
            Path or packaged states filename for the wind-direction CSV.
        height
            Height coordinate in metres assigned to both wind fields,
            regardless of the filenames or measurement-height metadata.
        extrapolation_heights
            Additional heights in metres to generate for wind speed with
            ``ABLLogWsProfile``. The reference ``height`` is always included.
        profile_z0
            Roughness length in metres. Required when an additional height
            differs from the reference height.
        profile_mol
            Monin-Obukhov length in metres. NaN selects neutral conditions.
        output_vars
            Requested variables from ``FV.WS`` and ``FV.WD``. Defaults to both.
        load_mode
            Must be ``preload`` because the CSV files are parsed into one
            in-memory dataset during construction.
        kwargs
            Additional ``LatLonFieldData`` and ``DatasetStates`` parameters.

        Raises
        ------
        FileNotFoundError
            If either input file is absent from both the configured input
            directory and the packaged states data. Existing local files take
            precedence; packaged files are looked up by basename.
        ValueError
            If their grid, states, or values are inconsistent, ``height`` or
            profile inputs are invalid, or ``load_mode`` is not ``preload``.
        """
        if load_mode != "preload":
            raise ValueError("PECDStates: load_mode must be 'preload'")
        variables = [FV.WS, FV.WD] if output_vars is None else list(output_vars)
        unsupported = set(variables) - {FV.WS, FV.WD}
        if unsupported:
            raise ValueError(
                f"PECDStates: Unsupported output variables {sorted(unsupported)}; "
                f"choose from {FV.WS}, {FV.WD}"
            )

        kwargs.setdefault("utm_zone", "from_grid")
        super().__init__(
            data_source=_read_pecd_dataset(
                wind_speed_file,
                wind_direction_file,
                height,
                extrapolation_heights,
                profile_z0,
                profile_mol,
            ),
            output_vars=variables,
            var2ncvar={FV.WS: FV.WS, FV.WD: FV.WD},
            load_mode=load_mode,
            states_coord=FC.STATE,
            lat_coord="latitude",
            lon_coord="longitude",
            h_coord="height",
            time_format=None,
            **kwargs,
        )
