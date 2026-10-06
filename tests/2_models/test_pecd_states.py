from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.input.states.pecd_states import PECDStates


_POINT_IDS = [1, 2, 3, 4]
_LATITUDES = [53.0, 53.0, 53.25, 53.25]
_LONGITUDES = [3.0, 3.25, 3.0, 3.25]


def _decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _write_pecd_file(
    path: Path,
    height: int,
    values: list[list[float]],
    latitudes: list[float] = _LATITUDES,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerows(
            [
                ["!DATEITYP", "test"],
                ["!!DATUM_ERSTELLUNG", "20261005"],
                ["!!VERSION_SCHNITTSTELLE", "14"],
                ["!LAND", *("DE" for _ in _POINT_IDS)],
                ["!MESSHOEHE_M", *(str(height) for _ in _POINT_IDS)],
                ["!BREITENGRAD_DEG_N", *map(_decimal, latitudes)],
                ["!LAENGENGRAD_DEG_E", *map(_decimal, _LONGITUDES)],
                ["NNF_ID", *map(str, _POINT_IDS)],
            ]
        )
        for state_index, row in enumerate(values, start=1):
            writer.writerow(
                [state_index, *(str(value).replace(".", ",") for value in row)]
            )


def _write_input_files(input_dir: Path, bad_grid: bool = False) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_pecd_file(
        input_dir / "speed_input.csv",
        100,
        [[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
    )
    _write_pecd_file(
        input_dir / "direction_input.csv",
        10,
        [[270.0, 271.0, 272.0, 273.0], [180.0, 181.0, 182.0, 183.0]],
        [53.0, 53.0, 53.5, 53.5] if bad_grid else _LATITUDES,
    )


def test_pecd_states_reads_explicit_files_at_default_height(tmp_path):
    _write_input_files(tmp_path)

    states = PECDStates(tmp_path / "speed_input.csv", tmp_path / "direction_input.csv")
    data = states.data_source

    assert data.sizes == {
        FC.STATE: 2,
        "height": 1,
        "latitude": 2,
        "longitude": 2,
    }
    np.testing.assert_array_equal(data[FC.STATE], [1, 2])
    np.testing.assert_array_equal(data["height"], [100.0])
    np.testing.assert_array_equal(
        data[FV.WS].isel({FC.STATE: 0, "height": 0}), [[11, 12], [13, 14]]
    )
    np.testing.assert_array_equal(
        data[FV.WD].isel({FC.STATE: 0, "height": 0}), [[270, 271], [272, 273]]
    )


def test_pecd_states_reads_packaged_ten_day_subset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(config, FC.INPUT_DIR, tmp_path)
    data = PECDStates("pecd_wind_speed.csv", "pecd_wind_direction.csv").data_source

    assert data.sizes[FC.STATE] == 240
    assert data.sizes["latitude"] == 13
    assert data.sizes["longitude"] == 13
    np.testing.assert_array_equal(data[FC.STATE][[0, -1]], [1, 240])
    np.testing.assert_array_equal(data["longitude"], np.arange(3.0, 6.01, 0.25))


def test_pecd_states_prefers_configured_input_files(tmp_path, monkeypatch):
    _write_input_files(tmp_path)
    (tmp_path / "speed_input.csv").rename(tmp_path / "pecd_wind_speed.csv")
    (tmp_path / "direction_input.csv").rename(tmp_path / "pecd_wind_direction.csv")
    monkeypatch.setitem(config, FC.INPUT_DIR, tmp_path)

    data = PECDStates("pecd_wind_speed.csv", "pecd_wind_direction.csv").data_source

    assert data.sizes[FC.STATE] == 2
    np.testing.assert_array_equal(
        data[FV.WS].isel({FC.STATE: 0, "height": 0}), [[11, 12], [13, 14]]
    )
    np.testing.assert_array_equal(
        data[FV.WD].isel({FC.STATE: 0, "height": 0}), [[270, 271], [272, 273]]
    )


def test_pecd_states_profiles_wind_speed_to_additional_heights(tmp_path):
    _write_input_files(tmp_path)

    data = PECDStates(
        tmp_path / "speed_input.csv",
        tmp_path / "direction_input.csv",
        extrapolation_heights=[80.0, 120.0],
        profile_z0=0.05,
    ).data_source

    np.testing.assert_array_equal(data["height"], [80.0, 100.0, 120.0])
    reference = data[FV.WS].sel(height=100.0)
    np.testing.assert_array_equal(reference.isel({FC.STATE: 0}), [[11, 12], [13, 14]])
    assert np.all(data[FV.WS].sel(height=80.0) < reference)
    assert np.all(data[FV.WS].sel(height=120.0) > reference)
    np.testing.assert_array_equal(
        data[FV.WD].sel(height=80.0), data[FV.WD].sel(height=120.0)
    )


def test_pecd_states_requires_roughness_length_for_extrapolation(tmp_path):
    _write_input_files(tmp_path)

    with pytest.raises(ValueError, match="profile_z0 must be finite and positive"):
        PECDStates(
            tmp_path / "speed_input.csv",
            tmp_path / "direction_input.csv",
            extrapolation_heights=[120.0],
        )


def test_pecd_states_rejects_inconsistent_file_grids(tmp_path):
    _write_input_files(tmp_path, bad_grid=True)

    with pytest.raises(ValueError, match="Grid metadata differs"):
        PECDStates(tmp_path / "speed_input.csv", tmp_path / "direction_input.csv")


@pytest.mark.parametrize("variable", [FV.WS, FV.WD])
def test_pecd_states_rejects_missing_input_file(tmp_path, variable):
    _write_input_files(tmp_path)
    speed_file = tmp_path / "speed_input.csv"
    direction_file = tmp_path / "direction_input.csv"
    if variable == FV.WS:
        speed_file = tmp_path / "missing_speed.csv"
    else:
        direction_file = tmp_path / "missing_direction.csv"
    with pytest.raises(
        FileNotFoundError, match=f"{variable} file not found.*no packaged"
    ):
        PECDStates(speed_file, direction_file)
