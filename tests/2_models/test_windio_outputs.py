"""Local numerical regressions for WindIO constructor-defined outputs."""

from copy import deepcopy
from importlib import import_module

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.input.yaml.dict import run_outputs
from foxes.input.yaml.windio.read_outputs import read_outputs
from foxes.input.yaml.windio.read_site import get_states
from foxes.utils import Dict


@pytest.fixture
def algorithm():
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [6.0, 8.0, 10.0]},
            index=pd.date_range("2020", periods=3),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for index in range(2):
        farm.add_turbine(
            foxes.Turbine([index * 500.0, 0.0], turbine_models=["NREL5MW"], H=90.0),
            verbosity=0,
        )
    return foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)


def _request(family, fields, subset=(2, 0), **sampling):
    outputs = {
        "output_folder": "windio-results",
        "run_configuration": {
            "times_run": {
                "all_occurences": subset is None,
                "subset": None if subset is None else list(subset),
            }
        },
    }
    if family == "turbine":
        outputs["turbine_outputs"] = {
            "turbine_nc_filename": "selected-turbines.nc",
            "output_variables": fields,
        }
    else:
        outputs["flow_field"] = {
            "flow_nc_filename": "selected-flow.nc",
            "output_variables": fields,
            "z_planes": {
                "z_sampling": "plane_list",
                "z_list": [110.0, 90.0],
                "xy_sampling": "grid",
                "x_bounds": [-10.0, 110.0],
                "y_bounds": [-20.0, 20.0],
                "Nx": 3,
                "Ny": 2,
                **sampling,
            },
        }
    return Dict(outputs)


@pytest.fixture
def weibull_algorithm(algorithm):
    coords = {
        FV.WD: [90.0, 270.0],
        FV.WS: [6.0, 8.0, 10.0],
        FV.X: [-2000.0, 0.0, 2000.0, 4000.0, 6000.0],
        FV.Y: [-2000.0, 0.0, 2000.0],
        FV.H: [0.0, 30.0, 90.0, 200.0, 400.0],
    }
    field_dims = (FV.WD, FV.X, FV.Y, FV.H)
    fields = {
        FV.WEIBULL_A: np.full((2, 5, 3, 5), 8.0),
        FV.WEIBULL_k: np.full((2, 5, 3, 5), 2.0),
        "sector_probability": np.array([0.4, 0.6]),
        FV.TI: np.array([0.05, 0.06, 0.09, 0.14, 0.21]),
    }
    dims = {
        FV.WEIBULL_A: field_dims,
        FV.WEIBULL_k: field_dims,
        "sector_probability": (FV.WD,),
        FV.TI: (FV.X,),
    }
    algorithm.states = get_states(coords, fields, dims, verbosity=0)
    return algorithm


@pytest.mark.parametrize("engine_type", ["numpy", "process"])
def test_weibull_flow_retains_support_outside_farm(
    weibull_algorithm, tmp_path, monkeypatch, engine_type
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    request = _request(
        "flow",
        ["wind_speed", "wind_direction"],
        subset=None,
        x_bounds=[-1000.0, 5000.0],
        y_bounds=[-1000.0, 1000.0],
        z_list=[350.0, 90.0],
    )
    definition = Dict()
    read_outputs(request, definition, weibull_algorithm, verbosity=0)
    source = weibull_algorithm.states.data_source.copy(deep=True)
    with foxes.Engine.new(
        engine_type,
        n_procs=2,
        chunk_size_states=4,
        chunk_size_points=5,
        verbosity=0,
    ):
        farm = weibull_algorithm.calc_farm()
        results, _ = run_outputs(
            definition, weibull_algorithm, farm, with_engine=True, verbosity=0
        )
        points = weibull_algorithm.calc_points(
            farm,
            np.array([[-2000.0, 0.0, 90.0], [5000.0, 0.0, 90.0], [6000.0, 0.0, 90.0]]),
        )
    data = results[0][1][0]
    assert set(data.dims) == {"z", "time", "y", "x"}
    assert data.wind_speed.dims == ("z", "time", "y", "x")
    np.testing.assert_allclose(data.x, [-1000.0, 2000.0, 5000.0])
    np.testing.assert_allclose(data.y, [-1000.0, 1000.0])
    np.testing.assert_allclose(data.z, [350.0, 90.0])
    np.testing.assert_allclose(
        data.wind_speed.isel(z=0, y=0, x=-1),
        farm[FV.AMB_REWS].isel({FC.TURBINE: 0}),
    )
    np.testing.assert_allclose(
        data.wind_direction.isel(z=0, y=0, x=-1),
        farm[FV.WD].isel({FC.TURBINE: 0}),
    )
    np.testing.assert_allclose(
        points[FV.TI], np.tile([0.05, 0.175, 0.21], (farm.sizes[FC.STATE], 1))
    )
    with xr.open_dataset(tmp_path / "selected-flow.nc") as written:
        xr.testing.assert_allclose(data, written)
    xr.testing.assert_identical(weibull_algorithm.states.data_source, source)


@pytest.mark.parametrize("point", [[6001.0, 0.0, 90.0], [0.0, 0.0, 401.0]])
def test_weibull_points_outside_source_still_raise(weibull_algorithm, point):
    with foxes.Engine.new("numpy", verbosity=0):
        farm = weibull_algorithm.calc_farm()
        with pytest.raises(ValueError, match="out of bounds"):
            weibull_algorithm.calc_points(farm, np.array([point]))


@pytest.mark.parametrize("aggregate", [None, "wind_farm", "cluster", "all"])
@pytest.mark.parametrize("mean", [False, True])
def test_turbine_aggregation_preselects_states_and_preserves_windio_names(
    algorithm, tmp_path, monkeypatch, aggregate, mean
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    request = _request("turbine", ["power", "rotor_effective_velocity"])
    request["turbine_outputs"].update(
        aggregate_turbines=aggregate, aggregate_states=mean
    )
    definition = Dict()
    read_outputs(request, definition, algorithm, verbosity=0)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        result, _ = run_outputs(
            definition, algorithm, farm, with_engine=True, verbosity=0
        )
    data = result[0][1][0]
    for external, variable in [("power", FV.P), ("rotor_effective_velocity", FV.REWS)]:
        expected = farm[variable].isel({FC.STATE: [2, 0]})
        if mean:
            expected = expected.mean(FC.STATE)
        if aggregate:
            expected = (
                expected.sum(FC.TURBINE)
                if variable == FV.P
                else expected.mean(FC.TURBINE)
            )
        np.testing.assert_allclose(data[external].squeeze(), expected.T.squeeze())
    assert ("time" in data.dims) is (not mean)
    assert data.attrs.get("n_states") == (2 if mean else None)
    assert data.attrs.get("n_turbines") == (2 if aggregate else None)
    if aggregate in ("wind_farm", "cluster"):
        assert aggregate in data.dims
    with xr.open_dataset(tmp_path / "selected-turbines.nc") as written:
        xr.testing.assert_allclose(data, written)
        assert data.attrs == written.attrs


def test_flow_state_aggregation_normalizes_partial_selection_and_directions(
    algorithm, tmp_path, monkeypatch
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    algorithm.states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [6.0, 8.0, 10.0], FV.WD: [350.0, 90.0, 10.0]},
            index=pd.date_range("2020", periods=3),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
    )
    request = _request("flow", ["wind_speed", "wind_direction"])
    request["flow_field"]["aggregate_states"] = True
    definition = Dict()
    read_outputs(request, definition, algorithm, verbosity=0)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=3, verbosity=0
    ):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        result, _ = run_outputs(
            definition, algorithm, farm, with_engine=True, verbosity=0
        )
    data = result[0][1][0]
    assert set(data.dims) == {"x", "y", "z"}
    assert data.attrs["n_states"] == 2
    assert data.attrs["aggregate_states"] == 1
    np.testing.assert_allclose(data.wind_speed, 8.0)
    np.testing.assert_allclose(
        np.exp(1j * np.deg2rad(data.wind_direction)), 1.0, atol=1e-12
    )
    with xr.open_dataset(tmp_path / "selected-flow.nc") as written:
        xr.testing.assert_identical(data, written)


@pytest.mark.parametrize(
    "family,options,message",
    [
        ("flow", {"aggregate_points": "all"}, "remain gridded"),
        ("flow", {"aggregate_states": "true"}, "boolean"),
        ("turbine", {"aggregate_states": 1}, "boolean"),
        ("turbine", {"aggregate_turbines": "unknown"}, "aggregate_turbines"),
    ],
)
def test_windio_rejects_invalid_aggregation(algorithm, family, options, message):
    request = _request(family, ["power"] if family == "turbine" else ["wind_speed"])
    request["turbine_outputs" if family == "turbine" else "flow_field"].update(options)
    with pytest.raises(ValueError, match=message):
        read_outputs(request, Dict(), algorithm, verbosity=0)


def _assert_selected_windio_fields(
    dataset, farm, family, fields, foxes_fields, positions
):
    """Verify physical values, external axes and requested ordering for either family."""
    np.testing.assert_array_equal(
        dataset.time, farm[FC.STATE].isel({FC.STATE: positions})
    )
    if family == "turbine":
        assert all(dataset[field].dims == ("turbine", "time") for field in fields)
        for field, foxes_field in zip(fields, foxes_fields):
            np.testing.assert_allclose(
                dataset[field], farm[foxes_field].isel({FC.STATE: positions}).T
            )
    else:
        assert all(dataset[field].dims == ("z", "time", "y", "x") for field in fields)
        np.testing.assert_array_equal(dataset.z, [110.0, 90.0])
        np.testing.assert_allclose(dataset.x, [-10.0, 50.0, 110.0])
        np.testing.assert_allclose(dataset.y, [-20.0, 20.0])
        for field in fields:
            expected = (
                np.asarray([6.0, 8.0, 10.0])[positions]
                if field == "wind_speed"
                else np.full(len(positions), 270.0)
            )
            np.testing.assert_allclose(dataset[field].isel(z=0, y=0, x=0), expected)


@pytest.mark.parametrize(
    "family,fields,foxes_fields",
    [
        ("turbine", ["power"], [FV.P]),
        ("turbine", ["rotor_effective_velocity"], [FV.REWS]),
        ("turbine", ["power", "rotor_effective_velocity"], [FV.P, FV.REWS]),
        ("flow", ["wind_speed"], [FV.WS]),
        ("flow", ["wind_direction"], [FV.WD]),
        ("flow", ["wind_speed", "wind_direction"], [FV.WS, FV.WD]),
    ],
)
@pytest.mark.parametrize(
    "subset,verbosity,nofig", [((2, 0), 0, True), (None, 1, False)]
)
def test_selected_fields_execute_and_write(
    algorithm,
    tmp_path,
    monkeypatch,
    capsys,
    family,
    fields,
    foxes_fields,
    subset,
    verbosity,
    nofig,
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    definition = Dict()
    directory = read_outputs(
        _request(family, fields, subset=subset),
        definition,
        algorithm,
        verbosity=verbosity,
    )
    entry = definition["outputs"][0]
    assert directory == "windio-results"
    assert entry["variables"] == foxes_fields
    assert "functions" not in entry

    execute = algorithm.calc_outputs
    executed = []

    def execute_output(*args, **kwargs):
        output = kwargs["outputs"]
        executed.append((type(output).__name__, output.nofig))
        return execute(*args, **kwargs)

    monkeypatch.setattr(algorithm, "calc_outputs", execute_output)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=3, verbosity=0
    ):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        results, _ = run_outputs(
            definition,
            algorithm,
            farm,
            with_engine=True,
            nofig=nofig,
            verbosity=verbosity,
        )
    dataset = results[0][1][0]
    output_type = "StateTurbineTable" if family == "turbine" else "SlicesData"
    assert executed == [(output_type, nofig)]
    assert set(dataset.data_vars) == set(fields)
    positions = [0, 1, 2] if subset is None else list(subset)
    _assert_selected_windio_fields(
        dataset, farm, family, fields, foxes_fields, positions
    )

    filename = "selected-turbines.nc" if family == "turbine" else "selected-flow.nc"
    assert [path.name for path in tmp_path.glob("*.nc")] == [filename]
    with xr.open_dataset(tmp_path / filename) as written:
        xr.testing.assert_allclose(written, dataset)
    text = capsys.readouterr().out
    assert ("Writing file" in text) == (verbosity > 0)
    if verbosity == 0:
        assert text == ""


def test_windio_entrypoint_forwards_nofig(algorithm, monkeypatch, tmp_path):
    windio = import_module("foxes.input.yaml.windio.windio")
    received = {}
    monkeypatch.setattr(
        windio, "read_windio_file", lambda *args, **kwargs: (Dict(), algorithm, ".")
    )

    def run(*args, **kwargs):
        received.update(kwargs)
        return "executed"

    monkeypatch.setattr(windio, "run_dict", run)
    result = windio.foxes_windio(tmp_path / "synthetic.yaml", nofig=True, verbosity=0)
    assert result == "executed"
    assert received["nofig"] is True
    assert received["verbosity"] == 0


def test_entrypoint_runs_both_actual_output_families(
    algorithm, monkeypatch, tmp_path, capsys
):
    windio = import_module("foxes.input.yaml.windio.windio")
    request = _request("turbine", ["power"])
    request["flow_field"] = _request("flow", ["wind_speed"])["flow_field"]
    definition = Dict(calc_farm=Dict(run=True))
    read_outputs(request, definition, algorithm, verbosity=0)
    monkeypatch.setitem(config, FC.INPUT_DIR, tmp_path)
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    monkeypatch.setattr(
        windio,
        "read_windio_file",
        lambda *args, **kwargs: (definition, algorithm, "unused-output-folder"),
    )
    result = windio.foxes_windio(
        tmp_path / "synthetic.yaml",
        output_dir=tmp_path,
        engine="numpy",
        chunksize_states=2,
        chunksize_points=3,
        nofig=True,
        verbosity=0,
    )
    table, flow = [entry[1][0] for entry in result[1:]]
    np.testing.assert_array_equal(table.time, flow.time)
    assert set(table.data_vars) == {"power"}
    assert set(flow.data_vars) == {"wind_speed"}
    assert {path.name for path in tmp_path.glob("*.nc")} == {
        "selected-turbines.nc",
        "selected-flow.nc",
    }
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("family", ["turbine", "flow"])
def test_disabled_output_does_not_require_constructor_options(algorithm, family):
    key = "turbine_outputs" if family == "turbine" else "flow_field"
    definition = Dict()
    assert read_outputs(Dict({key: {"report": False}}), definition, algorithm, 0) == "."
    assert "outputs" not in definition


@pytest.mark.parametrize("family", ["turbine", "flow"])
def test_unknown_selected_fields_are_rejected(algorithm, family):
    with pytest.raises(KeyError, match="unsupported"):
        read_outputs(_request(family, ["unsupported"]), Dict(), algorithm, 0)


@pytest.mark.parametrize(
    "sampling,exception,message",
    [
        ({"xy_sampling": "unsupported"}, NotImplementedError, "xy_sampling"),
        ({"z_sampling": "unsupported"}, NotImplementedError, "z_sampling"),
    ],
)
def test_unsupported_sampling_is_rejected(algorithm, sampling, exception, message):
    with pytest.raises(exception, match=message):
        read_outputs(_request("flow", ["wind_speed"], **sampling), Dict(), algorithm, 0)


def test_missing_grid_counts_are_rejected(algorithm):
    request = _request("flow", ["wind_speed"])
    planes = request["flow_field"]["z_planes"]
    planes.pop("Nx")
    planes.pop("Ny")
    with pytest.raises(KeyError, match="dx.*dy.*Nx.*Ny"):
        read_outputs(request, Dict(), algorithm, 0)


@pytest.mark.parametrize("subset", [[], [-1], [True], [0.5], [1, 1]])
@pytest.mark.parametrize("family", ["turbine", "flow"])
def test_invalid_state_subsets_are_rejected(algorithm, subset, family):
    with pytest.raises(ValueError, match="state subset"):
        read_outputs(
            _request(
                family,
                ["power" if family == "turbine" else "wind_speed"],
                subset=subset,
            ),
            Dict(),
            algorithm,
            0,
        )


@pytest.mark.parametrize("family", ["turbine", "flow"])
def test_empty_field_selection_is_rejected(algorithm, tmp_path, monkeypatch, family):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    definition = Dict()
    request = _request(family, [])
    read_outputs(request, definition, algorithm, 0)
    with foxes.Engine.new("numpy", verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        with pytest.raises(ValueError, match="variables must contain"):
            run_outputs(definition, algorithm, farm, with_engine=True, verbosity=0)


@pytest.mark.parametrize(
    "sampling,expected",
    [
        ({"z_sampling": "hub_heights"}, [90.0]),
        (
            {"z_sampling": "grid", "z_bounds": [80.0, 100.0], "z_number": 3},
            [80.0, 90.0, 100.0],
        ),
        ({"z_sampling": 95.0}, [95.0]),
    ],
)
def test_flow_height_sampling(algorithm, tmp_path, monkeypatch, sampling, expected):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    definition = Dict()
    read_outputs(_request("flow", ["wind_speed"], **sampling), definition, algorithm, 0)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        results, _ = run_outputs(
            definition, algorithm, farm, with_engine=True, nofig=True, verbosity=0
        )
    np.testing.assert_array_equal(results[0][1][0].z, expected)


def test_spacing_grid_retains_bounds_and_requested_maximum_spacing(algorithm):
    request = _request("flow", ["wind_speed"], dx=50.0, dy=30.0)
    planes = request["flow_field"]["z_planes"]
    planes.pop("Nx")
    planes.pop("Ny")
    definition = Dict()
    read_outputs(request, definition, algorithm, verbosity=3)
    grid = definition["outputs"][0]["grid_pars"]
    assert grid["n_img_points"] == (4, 3)
    assert (grid["xmin"], grid["xmax"]) == (-10.0, 110.0)
    assert (grid["ymin"], grid["ymax"]) == (-20.0, 20.0)


def test_definition_can_be_executed_without_consuming_it(
    algorithm, tmp_path, monkeypatch
):
    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    definition = Dict()
    read_outputs(_request("turbine", ["power"]), definition, algorithm, 0)
    original = deepcopy(definition)
    with foxes.Engine.new("numpy", verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all")
        run_outputs(
            definition, algorithm, farm, with_engine=True, nofig=True, verbosity=0
        )
    assert definition == original
