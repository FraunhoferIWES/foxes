"""Contract, numerical and compact-payload checks for the final output adoption."""

import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.input.yaml.dict import read_dict, run_outputs
from foxes.models.outputs import (
    FarmLayoutOutput,
    FarmResultsChunkWriter,
    FarmResultsCountWriter,
    FarmResultsPeriodWriter,
    PointResultsCountWriter,
    PointResultsPeriodWriter,
    FlowAnimation,
    FlowPlots2D,
    FlowTurbineAnimation,
    PointCalculator,
    RosePlotOutput,
    RotorPointPlot,
    SliceData,
    SlicesData,
    StateTurbineMap,
    StateTurbineTable,
    StatesRosePlotOutput,
    TurbineTypeCurves,
    WindRoseBinPlot,
)
from foxes.utils import Dict


@pytest.mark.parametrize("configured", [None, 2])
def test_yaml_engine_inherits_quiet_level_without_overriding_options(
    algorithm, configured
):
    options = {} if configured is None else {"verbosity": configured}
    definition = Dict(engine=Dict(engine_type="numpy", **options))
    _, engine = read_dict(definition, algo=algorithm, verbosity=0)
    assert engine.verbosity == (0 if configured is None else configured)
    assert "verbosity" not in definition


@pytest.mark.parametrize(
    "cls,parameters",
    [
        (StateTurbineTable, {"to_file": "data.nc"}),
        (PointCalculator, {"to_file": "data.nc", "points": [[0, 0, 90]]}),
        (SliceData, {"to_file": "data.nc", "data_format": "xarray"}),
        (SlicesData, {"to_file": "data.nc", "positions": [80, 90]}),
        (FarmResultsCountWriter, {"base_name": "data", "n_states_per_file": 2}),
        (FarmResultsChunkWriter, {"base_name": "data"}),
        (FarmResultsPeriodWriter, {"base_name": "data", "period": "day"}),
    ],
)
@pytest.mark.parametrize("explicit", [None, 1])
def test_netcdf_writers_inherit_verbosity_and_preserve_explicit_choice(
    algorithm, tmp_path, capsys, cls, parameters, explicit
):
    algorithm.states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [6, 8, 10, 12, 14]},
            index=pd.date_range("2020", periods=5),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    options = {} if explicit is None else {"verbosity": explicit}
    key = "write_nc_pars" if "base_name" in parameters else "write_pars"
    output = cls(**parameters, **{key: options}, out_dir=tmp_path)
    if isinstance(output, SliceData):
        output.grid_pars = {"n_img_points": (2, 2)}
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=output)
    assert list(tmp_path.glob("*.nc"))
    text = capsys.readouterr().out
    assert ("Writing file" in text) == (explicit == 1)


@pytest.mark.parametrize("positional", [False, True])
def test_yaml_pyplot_savefig_resolves_output_directory(
    tmp_path, monkeypatch, positional
):
    from foxes.config import config

    monkeypatch.setitem(config, FC.OUTPUT_DIR, tmp_path)
    options = {"args": ["saved.png"]} if positional else {"fname": "saved.png"}
    definition = Dict(
        outputs=[
            Dict(
                output_type="plt",
                functions=[
                    Dict(function="figure", result_labels="$fig"),
                    Dict(function="savefig", **options),
                    Dict(function="close", args=["$fig"]),
                ],
            )
        ]
    )
    results, storage = run_outputs(definition, verbosity=0)
    assert (tmp_path / "saved.png").stat().st_size
    assert storage["$fig"].number not in plt.get_fignums()
    assert len(results[0][1]) == 3


@pytest.mark.parametrize("name", ["does_not_exist", "_get_backend_mod", "rcParams"])
def test_yaml_pyplot_rejects_nonpublic_or_noncallable_functions(name):
    definition = Dict(
        outputs=[Dict(output_type="plt", functions=[Dict(function=name)])]
    )
    with pytest.raises(ValueError, match="public Matplotlib function"):
        run_outputs(definition, verbosity=0)


def test_yaml_pyplot_requires_matplotlib_objects_and_matching_phase():
    definition = Dict(
        outputs=[Dict(output_type="plt", functions=[Dict(function="figure")])]
    )
    results, _ = run_outputs(definition, with_engine=True, verbosity=0)
    assert results[0][1] is None
    with pytest.raises(TypeError, match="Matplotlib artist"):
        run_outputs(
            Dict(outputs=[Dict(object="$data", functions=[])]),
            results_storage=Dict({"$data": xr.Dataset()}),
            verbosity=0,
        )


def test_table_name_map_can_include_unselected_fields(algorithm):
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        table = algorithm.calc_farm(
            outputs=StateTurbineTable(
                variables=[FV.P],
                name_map={FV.P: "power", FV.REWS: "speed", FC.STATE: "time"},
            )
        )
    assert set(table.data_vars) == {"power"}
    np.testing.assert_array_equal(table.time, [10, 20, 30, 40, 50])


@pytest.mark.parametrize(
    "points", [[], [[0, 1]], [[0, np.nan, 3]], np.zeros((2, 1, 3))]
)
def test_invalid_point_coordinates_fail_contextually(algorithm, points):
    with pytest.raises(ValueError, match="coordinates"):
        algorithm.calc_outputs(outputs=PointCalculator(points=points))


def test_point_mean_requires_positive_weights(algorithm):
    point_data = xr.Dataset(
        {
            FV.WS: ((FC.STATE, FC.POINT), np.ones((5, 1))),
            FV.X: (FC.POINT, [0.0]),
            FV.Y: (FC.POINT, [0.0]),
            FV.H: (FC.POINT, [90.0]),
        },
        coords={FC.STATE: [10, 20, 30, 40, 50], FC.POINT: [0]},
    )
    farm_data = xr.Dataset(
        {FV.WEIGHT: (FC.STATE, np.zeros(5))},
        coords={FC.STATE: point_data[FC.STATE], FC.TURBINE: [0, 1]},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        with pytest.raises(ValueError, match="positive total weight"):
            algorithm.calc_outputs(
                outputs=PointCalculator(states_mean=True),
                farm_results=farm_data,
                point_results=point_data,
            )


def _snapshot(output):
    return pickle.dumps(
        {
            key: value
            for key, value in vars(output).items()
            if key not in ("calculate", "_collect_points")
        }
    )


class _ParentOnlyFigure:
    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise AssertionError("parent-only figure reached worker serialization")


def _axis_title(index, state, *, ax):
    return ax.get_title() or f"State {state}"


def _configured_plot(output_type, fig, ax, **kwargs):
    constructors = {
        "layout": (FarmLayoutOutput, {"parameters": {"fig": fig, "ax": ax}}),
        "rose": (RosePlotOutput, {"plot_pars": {"ax": ax}}),
        "states_rose": (
            StatesRosePlotOutput,
            {"point": [0, 0, 90], "plot_pars": {"ax": ax}},
        ),
        "bins": (
            WindRoseBinPlot,
            {"variable": FV.P, "ws_bins": [0, 20], "plot_pars": {"ax": ax}},
        ),
        "rotor": (RotorPointPlot, {"plot_pars": {"fig": fig, "ax": ax}}),
        "map": (StateTurbineMap, {"plot_pars": {"ax": ax}}),
        "curves": (TurbineTypeCurves, {"plot_pars": {"axs": [ax, ax]}}),
        "flow": (
            FlowPlots2D,
            {
                "plot_pars": {"fig": fig, "ax": ax},
                "grid_pars": {"n_img_points": (2, 2)},
            },
        ),
        "animation": (
            FlowAnimation,
            {
                "plot_pars": {"title": partial(_axis_title, ax=ax)},
                "grid_pars": {"n_img_points": (2, 2)},
            },
        ),
        "turbine_animation": (
            FlowTurbineAnimation,
            {
                "plot_pars": {"title": partial(_axis_title, ax=ax)},
                "grid_pars": {"n_img_points": (2, 2)},
            },
        ),
    }
    cls, parameters = constructors[output_type]
    return cls(**parameters, **kwargs)


@pytest.mark.parametrize(
    "output_type",
    [
        "layout",
        "rose",
        "states_rose",
        "bins",
        "rotor",
        "map",
        "curves",
        "flow",
        "animation",
        "turbine_animation",
    ],
)
def test_plot_configuration_stays_out_of_running_models(
    algorithm, tmp_path, output_type
):
    # A non-serializable figure graph exposes even indirect worker references.
    fig, ax = plt.subplots()
    fig._foxes_parent_only = _ParentOnlyFigure()
    output = _configured_plot(output_type, fig, ax, out_dir=tmp_path)
    stash = {}
    try:
        algorithm.initialize()
        algorithm.init_model(output)
        initial = output.get_initial_output_results(algorithm)
        original = {output.unvar(key): value for key, value in initial.items()}
        assert all(getattr(output, name) is value for name, value in original.items())
        with foxes.Engine.new("numpy", verbosity=0):
            output.set_running(algorithm, stash)
            try:
                assert len(pickle.dumps(output)) < 32768
                assert all(not hasattr(output, name) for name in original)
            finally:
                output.unset_running(algorithm, stash)
        assert all(getattr(output, name) is value for name, value in original.items())
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "output",
    [
        StateTurbineTable(name_map={f"unused_{i}": str(i) for i in range(5000)}),
        StateTurbineMap(isel={FC.STATE: np.zeros(20000, dtype=int)}),
        TurbineTypeCurves(sel={FC.STATE: np.zeros(20000, dtype=int)}),
        SliceData(label_map={f"unused_{i}": str(i) for i in range(5000)}),
        FlowPlots2D(normalize={f"unused_{i}": 1.0 for i in range(5000)}),
        FarmResultsCountWriter(
            "farm", 2, write_nc_pars={"engine": _ParentOnlyFigure()}
        ),
        FarmResultsPeriodWriter(
            "farm", period="day", write_nc_pars={"engine": _ParentOnlyFigure()}
        ),
        PointResultsCountWriter(
            "point", 2, write_nc_pars={"engine": _ParentOnlyFigure()}
        ),
        PointResultsPeriodWriter(
            "point", period="day", write_nc_pars={"engine": _ParentOnlyFigure()}
        ),
    ],
)
def test_parent_format_and_buffered_writer_options_are_not_serialized(
    algorithm, output
):
    # Large formatting dictionaries and parent-only writer options stay on the parent.
    algorithm.initialize()
    algorithm.init_model(output)
    initial = output.get_initial_output_results(algorithm)
    original = {output.unvar(key): value for key, value in initial.items()}
    stash = {}
    with foxes.Engine.new("numpy", verbosity=0):
        output.set_running(algorithm, stash)
        try:
            assert len(pickle.dumps(output)) < 32768
            assert all(not hasattr(output, key) for key in original)
        finally:
            output.unset_running(algorithm, stash)
    assert all(getattr(output, key) is value for key, value in original.items())


@pytest.mark.parametrize(
    "output",
    [
        FarmResultsCountWriter("farm", 2),
        FarmResultsPeriodWriter("farm", period="day"),
        PointResultsCountWriter("point", 2),
        PointResultsPeriodWriter("point", period="day"),
    ],
)
def test_empty_buffered_writer_consumes_private_initial_results(algorithm, output):
    # Empty runs do not write files or leak their parent-only configuration.
    initial = output.get_initial_output_results(algorithm)
    output.finalize_results(algorithm, 0, 0, initial)
    assert not initial


@pytest.mark.parametrize("backend", ["numpy", "process"])
@pytest.mark.parametrize(
    "output_type",
    ["layout", "rose", "states_rose", "bins", "rotor", "map", "curves", "flow"],
)
def test_parent_figures_are_reused_without_serialization(
    algorithm, tmp_path, backend, output_type
):
    # Real process dispatch must reuse the parent's axes, not a deserialized copy.
    polar = output_type in ("rose", "states_rose", "bins")
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"} if polar else {})
    fig._foxes_parent_only = _ParentOnlyFigure()
    output = _configured_plot(output_type, fig, ax, out_dir=tmp_path)
    configuration = (
        "parameters"
        if output_type == "layout"
        else "plot_options"
        if output_type == "flow"
        else "plot_pars"
    )
    original = getattr(output, configuration)
    try:
        with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
            results = algorithm.calc_farm(outputs=[output], ret_dict=True)
        assert set(results) == {output.result_name}
        result = results[output.result_name]
        if output_type == "flow":
            assert result is fig
        elif output_type == "rotor":
            assert result.axes is ax
        elif polar:
            assert result[0] is ax
        elif output_type == "curves":
            assert all(axis is ax for axis in result)
        else:
            assert result is ax
        assert getattr(output, configuration) is original
    finally:
        plt.close(fig)


@pytest.mark.parametrize("backend", ["numpy", "process"])
@pytest.mark.parametrize("failure", ["worker", "finalize"])
def test_plot_configuration_is_restored_on_failure(
    algorithm, tmp_path, backend, failure
):
    # Both scientific and rendering failures release parent-only configuration.
    fig, ax = plt.subplots()
    fig._foxes_parent_only = _ParentOnlyFigure()
    output = (
        WindRoseBinPlot("unavailable", [0, 20], plot_pars={"ax": ax}, out_dir=tmp_path)
        if failure == "worker"
        else RotorPointPlot(plot_pars={"ax": ax, "marker": "invalid"}, out_dir=tmp_path)
    )
    original = output.plot_pars
    try:
        with foxes.Engine.new(backend, n_procs=2, chunk_size_states=2, verbosity=0):
            with pytest.raises(
                KeyError if failure == "worker" else ValueError,
                match="unavailable" if failure == "worker" else "marker",
            ):
                algorithm.calc_farm(outputs=output)
        assert output.plot_pars is original and output.plot_pars["ax"] is ax
        assert not output.running and not algorithm.running
    finally:
        plt.close(fig)


@pytest.fixture
def algorithm():
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [6.0, 8.0, 10.0, 12.0, 14.0], FV.WEIGHT: [0.1, 0.2, 0.0, 0.3, 0.4]},
            index=[10, 20, 30, 40, 50],
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for index in range(2):
        farm.add_turbine(
            foxes.Turbine(
                [index * 500.0, 0.0],
                name=f"Turbine {index}",
                turbine_models=["NREL5MW"],
            ),
            verbosity=0,
        )
    return foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)


@pytest.mark.parametrize(
    "name",
    [
        "state_turbine_map",
        "state_turbine_table",
        "point_calculator",
        "rotor_point_plot",
        "turbine_type_curves",
    ],
)
def test_remaining_model_book_presets_execute(algorithm, name, tmp_path):
    output = algorithm.mbook.outputs[name]
    output.nofig = True
    output.show = False
    if name in ("state_turbine_map", "rotor_point_plot", "turbine_type_curves"):
        output.plot = False
    if name == "point_calculator":
        output.points = [[0.0, 0.0, 90.0]]
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_farm(outputs=name)
    assert isinstance(result, xr.Dataset)
    assert "plt" not in algorithm.mbook.outputs


@pytest.mark.parametrize("engine", ["numpy", "threads", "process"])
@pytest.mark.parametrize("entry", ["calc_farm", "calc_outputs", "calc_points"])
@pytest.mark.parametrize("mean", [False, True])
def test_point_entry_points_and_worker_reduction(
    algorithm, engine, entry, mean, monkeypatch
):
    points = np.array([[i * 100.0, 0.0, 90.0] for i in range(5)])
    output = PointCalculator(points=points, states_mean=mean, weight_turbine=1)
    original = output._collect_points
    payloads = []

    def collect(*args, **kwargs):
        before = _snapshot(output)
        payload = original(*args, **kwargs)
        assert _snapshot(output) == before
        if mean:
            assert payload["values"][FV.WS].ndim == 1
            assert payload["points"].shape[-1] == 3
            assert "states" not in payload and "weights" not in payload
        payloads.append(payload)
        return payload

    if engine != "process":
        monkeypatch.setattr(output, "_collect_points", collect)
    with foxes.Engine.new(
        engine, n_procs=2, chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        if entry == "calc_farm":
            result = algorithm.calc_farm(outputs=output)
        else:
            farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)
            if entry == "calc_outputs":
                result = algorithm.calc_outputs(outputs=output, farm_results=farm)
            else:
                result = algorithm.calc_points(farm, points=points, outputs=output)
    if engine != "process":
        assert payloads
    if mean:
        assert result[FV.WS].dims == (FC.POINT,)
        np.testing.assert_allclose(result[FV.WS], 11.4)
    else:
        assert result[FV.WS].dims == (FC.STATE, FC.POINT)
        np.testing.assert_allclose(result[FV.WS][:, 0], [6, 8, 10, 12, 14])
    assert result.x.dims == (FC.POINT,)
    np.testing.assert_allclose(result.x, points[:, 0])
    assert output.points is points


@pytest.mark.parametrize("mean", [False, True])
def test_dynamic_point_coordinates_and_file(algorithm, tmp_path, mean):
    points = np.zeros((5, 3, 3))
    points[..., 0] = np.arange(5)[:, None] * 100.0
    points[..., 2] = 90.0
    output = PointCalculator(
        points=points,
        states_mean=mean,
        to_file="points.nc",
        write_pars={"verbosity": 0},
        out_dir=tmp_path,
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = algorithm.calc_farm(outputs=output)
    if mean:
        np.testing.assert_allclose(result.x, 270.0)
    else:
        np.testing.assert_allclose(result.x, points[..., 0])
    with xr.open_dataset(tmp_path / "points.nc") as written:
        xr.testing.assert_allclose(written, result)


def test_precomputed_points_retain_order_and_ignore_zero_weight_nan(algorithm):
    farm = xr.Dataset(
        {
            FV.WEIGHT: (
                (FC.STATE, FC.TURBINE),
                np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 0.0], [1.0, 3.0], [1.0, 5.0]]),
            )
        },
        coords={FC.STATE: [10, 20, 30, 40, 50], FC.TURBINE: [0, 1]},
    )
    values = np.array([np.nan, 8.0, np.nan, 12.0, 14.0])[:, None] * np.ones((1, 3))
    points = xr.Dataset(
        {
            FV.WS: ((FC.STATE, FC.POINT), values),
            FV.X: (FC.POINT, [20.0, 0.0, 10.0]),
            FV.Y: (FC.POINT, [0.0, 0.0, 0.0]),
            FV.H: (FC.POINT, [90.0, 90.0, 90.0]),
        },
        coords={FC.STATE: farm[FC.STATE], FC.POINT: [7, 2, 5]},
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = algorithm.calc_outputs(
            outputs=PointCalculator(states_mean=True, weight_turbine=1),
            farm_results=farm,
            point_results=points,
        )
    np.testing.assert_allclose(result[FV.WS], 12.2)
    np.testing.assert_array_equal(result[FC.POINT], [7, 2, 5])
    np.testing.assert_allclose(result.x, [20.0, 0.0, 10.0])


def test_table_fields_transposition_selection_and_auto_evaluation(
    algorithm, monkeypatch
):
    output = StateTurbineTable(
        variables=[FV.EFF, FV.WEIGHT],
        name_map={FV.EFF: "efficiency", FC.STATE: "time"},
        transpose=True,
        isel={"time": [4, 0], FC.TURBINE: [1]},
    )
    original = output.calculate
    payloads = []

    def calculate(*args, **kwargs):
        before = _snapshot(output)
        result = original(*args, **kwargs)
        assert _snapshot(output) == before
        payload = result[output.var("data")]
        assert set(payload["data_vars"]) == {FV.EFF, FV.WEIGHT}
        payloads.append(payload)
        return result

    monkeypatch.setattr(output, "calculate", calculate)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_farm(outputs=output)
    assert payloads
    assert result.efficiency.dims == (FC.TURBINE, "time")
    np.testing.assert_array_equal(result.time, [50, 10])
    np.testing.assert_allclose(result.efficiency, 1.0)


@pytest.mark.parametrize("labels", [[42], pd.date_range("2020", periods=3), ["a", "b"]])
def test_maps_singletons_and_non_numeric_labels(algorithm, labels, tmp_path):
    values = np.arange(len(labels) * 2).reshape(len(labels), 2)
    data = xr.Dataset(
        {FV.P: ((FC.STATE, FC.TURBINE), values)},
        coords={FC.STATE: labels, FC.TURBINE: [0, 1]},
    )
    # Match the algorithm metadata to the supplied state labels.
    algorithm.states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: np.ones(len(labels))}, index=labels),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        ax = algorithm.calc_outputs(
            outputs=StateTurbineMap(to_file="map.png", out_dir=tmp_path),
            farm_results=data,
        )
    np.testing.assert_array_equal(
        np.asarray(ax.collections[0].get_array()).ravel(), values.T.ravel()
    )
    assert (tmp_path / "map.png").stat().st_size
    plt.close(ax.figure)


def test_metadata_only_rotor_output(algorithm, tmp_path):
    with foxes.Engine.new("numpy", verbosity=0):
        metadata = algorithm.calc_outputs(outputs=RotorPointPlot(plot=False))
        assert metadata.points.shape[-1] == 3
        np.testing.assert_allclose(metadata.weights.sum(), 1)
        figure, first = plt.subplots()
        image = algorithm.calc_outputs(
            outputs=RotorPointPlot(
                plot_pars={"ax": first},
                to_file="rotor.png",
                out_dir=tmp_path,
            )
        )
        assert image.axes is first
        plt.close(figure)
    assert figure.number not in plt.get_fignums()
    assert (tmp_path / "rotor.png").stat().st_size


def test_curves_collect_executing_scan_without_hidden_runs(algorithm, monkeypatch):
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)

        def forbidden(*args, **kwargs):
            pytest.fail("A curve output launched a hidden farm calculation")

        monkeypatch.setattr(algorithm, "calc_farm", forbidden)
        axes = algorithm.calc_outputs(
            outputs=TurbineTypeCurves(),
            farm_results=farm,
        )
    assert len(axes) == 2
    for ax in axes:
        assert len(ax.lines) == 2
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), farm[FV.P][:, 0])
    plt.close(axes[0].figure)


def test_yaml_matplotlib_composes_constructor_outputs(algorithm, tmp_path):
    config = Dict(
        outputs=[
            Dict(
                output_type="plt",
                functions=[Dict(function="subplots", result_labels=["$fig", "$ax"])],
            ),
            Dict(
                output_type="RotorPointPlot", with_engine=False, plot_pars={"ax": "$ax"}
            ),
            Dict(
                object="$fig",
                functions=[Dict(function="savefig", fname=tmp_path / "composed.png")],
            ),
            Dict(output_type="plt", functions=[Dict(function="close", args=["$fig"])]),
        ]
    )
    results, storage = run_outputs(config, algorithm, verbosity=0)
    assert results[1][1][0].axes is storage["$ax"]
    assert (tmp_path / "composed.png").stat().st_size
    assert storage["$fig"].number not in plt.get_fignums()
    with pytest.raises(ValueError, match="constructor options"):
        run_outputs(
            Dict(outputs=[Dict(output_type="StateTurbineMap", functions=[])]), algorithm
        )


@pytest.mark.parametrize(
    "factory,options,match",
    [
        (StateTurbineTable, {"variables": []}, "field names"),
        (PointCalculator, {"variables": []}, "field names"),
    ],
)
def test_invalid_constructor_options(factory, options, match):
    with pytest.raises(ValueError, match=match):
        factory(**options)
