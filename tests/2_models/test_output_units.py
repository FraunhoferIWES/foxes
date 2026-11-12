"""Output unit metadata, turbine-type conversion and serialization contracts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.models.outputs import (
    FarmResultsOutput,
    PointCalculator,
    PointResultsOutput,
    RotorPointPlot,
    RosePlotOutput,
    SliceData,
    SlicesData,
    StateTurbineTable,
    WindRoseBinPlot,
)
from foxes.models.outputs._units import _normalize_farm_power, _set_units


@pytest.fixture
def algorithm():
    """Provide a small public-turbine farm with deterministic synthetic states."""
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: [6.0, 8.0, 10.0]}, index=pd.date_range("2026", periods=3)),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for index in range(2):
        farm.add_turbine(
            foxes.Turbine([index * 600.0, 0.0], turbine_models=["NREL5MW"]),
            verbosity=0,
        )

    return foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)


@pytest.mark.parametrize("unit", ["W", "kW", "MW"])
def test_power_uses_turbine_model_unit_and_preserves_uniform_values(unit):
    """Keep uniform native values unchanged and label each power field."""
    algo = SimpleNamespace(
        farm_controller=SimpleNamespace(turbine_types=[SimpleNamespace(P_unit=unit)])
    )
    values = np.array([[2]], dtype=int)
    values.flags.writeable = False
    fields = [FV.P, FV.AMB_P, FV.CAP, FV.MAX_P]
    payload = {
        "coords": {FC.TURBINE: np.array([0])},
        "data_vars": {v: ((FC.STATE, FC.TURBINE), values) for v in fields},
        "attrs": {},
    }

    _normalize_farm_power(payload, algo)
    dataset = _set_units(xr.Dataset(**payload))

    for variable in fields:
        assert dataset[variable].attrs["units"] == unit
        np.testing.assert_array_equal(dataset[variable], values)
    assert values[0, 0] == 2


def test_smallest_power_unit_uses_original_turbine_identity_and_is_idempotent():
    """Convert selected turbine identities once, even when outputs are reused."""
    algo = SimpleNamespace(
        farm_controller=SimpleNamespace(
            turbine_types=[SimpleNamespace(P_unit=unit) for unit in ("MW", "W", "kW")]
        )
    )
    values = np.array([[2, 3], [4, 5]])
    values.flags.writeable = False
    payload = {
        "coords": {FC.TURBINE: np.array([2, 0])},
        "data_vars": {FV.P: ((FC.STATE, FC.TURBINE), values)},
        "attrs": {},
    }

    _normalize_farm_power(payload, algo)
    np.testing.assert_array_equal(payload["data_vars"][FV.P][1], values * [1000, 1e6])
    assert payload["attrs"]["power_unit"] == "W"

    previous = _set_units(xr.Dataset(**payload))
    _normalize_farm_power(payload, algo, previous)
    np.testing.assert_array_equal(payload["data_vars"][FV.P][1], previous[FV.P])


@pytest.mark.parametrize("units", [[], ["invalid"]])
def test_missing_or_invalid_turbine_units_fail_explicitly(units):
    """Reject unavailable turbine types and unknown native power units."""
    algo = SimpleNamespace(
        farm_controller=SimpleNamespace(
            turbine_types=[SimpleNamespace(P_unit=unit) for unit in units]
        )
    )
    payload = {
        "coords": {FC.TURBINE: np.array([0])},
        "data_vars": {FV.P: ((FC.TURBINE,), np.ones(1))},
        "attrs": {},
    }

    with pytest.raises(ValueError, match="power units|turbine types"):
        _normalize_farm_power(payload, algo)


def test_known_ambient_units_and_custom_metadata():
    """Label ordinary and ambient fields without replacing custom metadata."""
    expected = {
        FV.X: "m",
        FV.REWS3: "m/s",
        FV.TI: "1",
        FV.WD: "deg",
        FV.TKE: "m2/s2",
        FV.RHO: "kg/m3",
        FV.T: "K",
        FV.p: "Pa",
        FV.YLD: "GWh/a",
        FV.P75: "GWh/a",
        FV.P90: "GWh/a",
        FV.CAPF: "1",
    }
    expected.update(
        {
            FV.var2amb[variable]: unit
            for variable, unit in list(expected.items())
            if variable in FV.var2amb
        }
    )
    dataset = xr.Dataset({variable: ((), 1.0) for variable in expected})
    dataset["custom"] = xr.DataArray(
        2.0, attrs={"units": "custom", "long_name": "custom field"}
    )

    _set_units(dataset)

    assert {
        variable: dataset[variable].attrs["units"] for variable in expected
    } == expected
    assert dataset["custom"].attrs == {"units": "custom", "long_name": "custom field"}


@pytest.mark.parametrize("engine", ["numpy", "threads", "process"])
def test_collectors_tables_and_files_have_units(algorithm, tmp_path, engine):
    """Preserve variable units through collection, renaming and every farm writer."""
    outputs = [
        FarmResultsOutput(
            variables=[FV.P, FV.AMB_P, FV.CAP, FV.YLD, FV.REWS, FV.WEIGHT]
        ),
        StateTurbineTable(
            variables=[FV.P],
            name_map={FV.P: "power"},
            to_file="table.nc",
            out_dir=tmp_path,
            result_name="table",
        ),
        algorithm.mbook.outputs["farm_results_P_write_n2"],
        algorithm.mbook.outputs["farm_results_P_write_chunks"],
        algorithm.mbook.outputs["farm_results_P_write_day"],
    ]
    for index, output in enumerate(outputs[2:]):
        output.out_dir = tmp_path
        output.base_name = f"farm_{index}"

    with foxes.Engine.new(engine, n_procs=2, chunk_size_states=2, verbosity=0):
        results = algorithm.calc_farm(outputs=outputs)

    farm_results = results[FC.FARM_RESULTS]
    power_unit = algorithm.farm_controller.turbine_types[0].P_unit
    expected_units = {
        FV.P: power_unit,
        FV.AMB_P: power_unit,
        FV.CAP: power_unit,
        FV.YLD: "GWh/a",
        FV.REWS: "m/s",
        FV.WEIGHT: "1",
    }
    for variable, unit in expected_units.items():
        assert farm_results[variable].attrs["units"] == unit
    assert results["table"]["power"].attrs["units"] == power_unit

    files = list(tmp_path.glob("*.nc"))
    assert len(files) == 8
    for path in files:
        with xr.open_dataset(path) as dataset:
            variable = FV.P if FV.P in dataset else "power"
            assert dataset[variable].attrs["units"] == power_unit


@pytest.mark.parametrize("mean", [False, True])
def test_point_datasets_and_files_have_units(algorithm, tmp_path, mean):
    """Keep point fields and coordinates labelled before and after state means."""
    output = PointCalculator(
        points=[[0, 0, 90], [200, 0, 90]],
        variables=[FV.WS, FV.TI],
        states_mean=mean,
        to_file="points.nc",
        out_dir=tmp_path,
    )

    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        point_results = algorithm.calc_farm(outputs=output)

    assert point_results[FV.WS].attrs["units"] == "m/s"
    assert point_results[FV.TI].attrs["units"] == "1"
    assert all(point_results[axis].attrs["units"] == "m" for axis in "xyz")
    with xr.open_dataset(tmp_path / "points.nc") as written:
        assert written[FV.WS].attrs["units"] == "m/s"
        assert written.x.attrs["units"] == "m"


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_slice_units_survive_renaming_and_writing(algorithm, tmp_path, multiple, mean):
    """Retain field and axis units in renamed single-plane and multi-plane files."""
    slice_type = SlicesData if multiple else SliceData
    options = {"positions": [80, 90]} if multiple else {"data_format": "xarray"}
    output = slice_type(
        variables=[FV.WS],
        reduction="mean" if mean else "states",
        grid_pars={"n_img_points": (2, 2)},
        label_map={FV.WS: "speed", "x": "east", "z": "height"},
        to_file="slice.nc",
        out_dir=tmp_path,
        **options,
    )

    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        slice_results = algorithm.calc_farm(outputs=output)

    assert slice_results.speed.attrs["units"] == "m/s"
    assert slice_results.east.attrs["units"] == slice_results.y.attrs["units"] == "m"
    if multiple:
        assert slice_results.height.attrs["units"] == "m"
    with xr.open_dataset(tmp_path / "slice.nc") as written:
        assert written.speed.attrs["units"] == "m/s"


def test_roses_and_rotor_metadata_have_units(algorithm):
    """Distinguish percent frequencies, native power and dimensionless rotor data."""
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        results = algorithm.calc_farm(
            outputs=[
                RosePlotOutput(ws_bins=[0, 20], plot=False, result_name="rose"),
                WindRoseBinPlot(
                    variable=FV.P, ws_bins=[0, 20], plot=False, result_name="bins"
                ),
                RotorPointPlot(plot=False, result_name="rotor"),
            ]
        )

    rose = results["rose"]
    assert rose.frequency.attrs["units"] == "%"
    assert rose[FV.AMB_WD].attrs["units"] == "deg"
    assert rose.bin_min_AMB_WD.attrs["units"] == "deg"
    assert results["bins"][FV.P].attrs["units"] == "kW"
    rotor = results["rotor"]
    assert rotor.points.attrs["units"] == rotor.weights.attrs["units"] == "1"


def test_point_collector_and_all_writer_policies(algorithm, tmp_path):
    """Label point geometry and wind fields in collected and serialized tiles."""
    outputs = [PointResultsOutput()]
    for suffix in ("write_n2", "write_chunks", "write_day"):
        output = algorithm.mbook.outputs[f"point_results_{suffix}"]
        output.out_dir = tmp_path
        outputs.append(output)

    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=1, verbosity=0
    ):
        farm_results = algorithm.calc_farm()
        point_results = algorithm.calc_points(
            farm_results, np.array([[0, 0, 90], [100, 0, 90]]), outputs=outputs
        )[FC.POINT_RESULTS]

    assert point_results[FV.WS].attrs["units"] == "m/s"
    assert point_results[FV.X].attrs["units"] == "m"
    files = list(tmp_path.glob("*.nc"))
    assert files
    for path in files:
        with xr.open_dataset(path) as written:
            assert written[FV.WS].attrs["units"] == "m/s"
            assert written[FV.H].attrs["units"] == "m"


def test_normalized_slice_coordinates_and_fields_are_dimensionless(algorithm):
    """Keep normalized slice coordinates consistent with dimensionless metadata."""
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        slice_results = algorithm.calc_farm(
            outputs=SlicesData(
                positions=[80, 90],
                variables=[FV.WS],
                grid_pars={"n_img_points": (2, 2)},
                normalize={"x": 100, "z": 100},
                normalize_v={FV.WS: 10},
            )
        )

    np.testing.assert_array_equal(slice_results.z, [0.8, 0.9])
    assert (
        slice_results.x.attrs["units"]
        == slice_results.z.attrs["units"]
        == slice_results[FV.WS].attrs["units"]
        == "1"
    )
    assert slice_results.y.attrs["units"] == "m"


def test_sequential_accumulation_preserves_variable_units(algorithm):
    """Carry per-step unit attributes into Sequential's accumulated datasets."""
    sequential = foxes.algorithms.Sequential(
        algorithm.farm,
        algorithm.states,
        wake_models=[],
        verbosity=0,
        points=np.array([[0, 0, 90]]),
    )

    with foxes.Engine.new("numpy", verbosity=0):
        for farm_results, point_results in sequential:
            assert farm_results[FV.P].attrs["units"] == "kW"
            assert point_results[FV.WS].attrs["units"] == "m/s"

        assert sequential.farm_results[FV.P].attrs["units"] == "kW"
        assert sequential.point_results[FV.WS].attrs["units"] == "m/s"
        assert sequential.point_results[FV.X].attrs["units"] == "m"
