"""Unified output execution with constructor-defined, run-local scientific data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.input.yaml.dict import run_outputs
from foxes.models.outputs import (
    FarmResultsOutput,
    FlowAnimation,
    FlowPlots2D,
    PointResultsOutput,
    SliceData,
    SlicesData,
    StatesRosePlotOutput,
    _point_calculation,
)
from foxes.utils import Dict


@pytest.fixture
def algorithm():
    # Synthetic weighted states give exact state and mean reference fields.
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {FV.WS: [6.0, 8.0, 10.0], FV.WEIGHT: [0.2, 0.3, 0.5]},
            index=[10, 20, 30],
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine([0.0, 0.0], H=90.0, D=100.0, turbine_models=["null_type"]),
        verbosity=0,
    )
    return foxes.algorithms.Downwind(
        farm, states, wake_models=[], rotor_model="centre", verbosity=0
    )


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("entry", ["calc_farm", "calc_outputs", "calc_points"])
def test_slice_execution_contract(algorithm, plane, entry, monkeypatch):
    output = SliceData(
        plane=plane, variables=[FV.WS], grid_pars={"n_img_points": (2, 3)}
    )
    blocks = []
    original = _point_calculation.point_blocks

    def bounded_blocks(*args, **kwargs):
        for targets in original(*args, **kwargs):
            blocks.append(targets.n_targets)
            yield targets

    monkeypatch.setattr(_point_calculation, "point_blocks", bounded_blocks)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        if entry == "calc_farm":
            result = algorithm.calc_farm(outputs=output)
            assert blocks and max(blocks) == 2
        else:
            farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)
            if entry == "calc_outputs":
                result = algorithm.calc_outputs(outputs=output, farm_results=farm)
            else:
                algorithm.init_model(output)
                points = output.get_calc_points(algorithm).copy()[::-1]

                def forbidden_points(_):
                    raise AssertionError("calc_points must ignore output grids")

                monkeypatch.setattr(output, "get_calc_points", forbidden_points)
                result = algorithm.calc_points(farm, points=points, outputs=output)
            assert not blocks
    assert result[FV.WS].dims == (FC.STATE, plane[1], plane[0])
    np.testing.assert_array_equal(result[FC.STATE], [10, 20, 30])
    np.testing.assert_allclose(result[FV.WS].values[:, 0, 0], [6.0, 8.0, 10.0])
    assert result.sizes[plane[0]] == 2
    assert result.sizes[plane[1]] == 3


def test_precomputed_points_skip_science_and_requested_grids(algorithm, monkeypatch):
    output = SliceData(
        variables=[FV.WS], reduction="mean", grid_pars={"n_img_points": (2, 3)}
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)
        algorithm.init_model(output)
        points = output.get_calc_points(algorithm)
        results = algorithm.calc_points(
            farm, points=points, outputs=PointResultsOutput(), finalize=False
        )
        results[FV.WS] = xr.full_like(results[FV.WS], 17.0)

        def forbidden(*args, **kwargs):
            raise AssertionError("Precomputed point results must not be recalculated")

        monkeypatch.setattr(output, "get_calc_points", forbidden)
        monkeypatch.setattr(algorithm.states, "calculate", forbidden)
        returned = algorithm.calc_outputs(
            outputs=output, farm_results=farm, point_results=results
        )
    np.testing.assert_allclose(returned[FV.WS], 17.0)


@pytest.mark.parametrize("entry", ["calc_farm", "calc_outputs"])
def test_disagreeing_output_grids_raise(algorithm, entry):
    outputs = [
        SliceData(grid_pars={"n_img_points": (2, 3)}),
        SliceData(grid_pars={"n_img_points": (3, 2)}),
    ]
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(ValueError, match="calculation points|Calculation points"):
            getattr(algorithm, entry)(outputs=outputs)


def test_output_selectors_remain_strings_and_lists(algorithm):
    algorithm.mbook.outputs["mean_speed"] = SliceData(
        variables=[FV.WS],
        reduction="mean",
        grid_pars={"n_img_points": (2, 3)},
        result_name="mean_speed",
    )
    with foxes.Engine.new("numpy", verbosity=0):
        result = algorithm.calc_farm(outputs=["farm_results_all", "mean_speed"])
    assert set(result) == {FC.FARM_RESULTS, "mean_speed"}
    np.testing.assert_allclose(result["mean_speed"][FV.WS], 8.6)


@pytest.mark.parametrize("algorithm_type", ["Downwind", "Iterative"])
@pytest.mark.parametrize("selector", ["name", "list", "instance", "chain"])
@pytest.mark.parametrize("ret_dict", [False, True])
def test_farm_result_mapping_is_opt_in(algorithm, algorithm_type, selector, ret_dict):
    # Keep ordinary returns unchanged and preserve the actual grouped key on request.
    algorithm.farm.turbines[0].models = ["NREL5MW"]
    algo = getattr(foxes.algorithms, algorithm_type)(
        algorithm.farm, algorithm.states, wake_models=[], verbosity=0
    )
    output = algo.mbook.outputs["total_results_mean"]
    outputs = {
        "name": "total_results_mean",
        "list": ["total_results_mean"],
        "instance": output,
        "chain": foxes.core.OutputList(outputs=[output]),
    }[selector]
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        returned = algo.calc_farm(outputs=outputs, ret_dict=ret_dict)
    if ret_dict:
        assert set(returned) == {FC.MEAN_TOTAL_FARM_RESULTS}
        returned = returned[FC.MEAN_TOTAL_FARM_RESULTS]
    assert isinstance(returned, xr.Dataset)
    np.testing.assert_allclose(returned[FV.REWS], 8.6)


@pytest.mark.parametrize("entry", ["calc_points", "calc_outputs"])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("ret_dict", [False, True])
def test_point_result_mapping_survives_delegation(
    algorithm, entry, precomputed, ret_dict
):
    # Exercise both calc_outputs delegation branches and direct point execution.
    output = SliceData(
        variables=[FV.WS], result_name="speed", grid_pars={"n_img_points": (2, 3)}
    )
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        algorithm.initialize()
        algorithm.init_model(output)
        points = output.get_calc_points(algorithm).copy()
        options = {"outputs": output, "ret_dict": ret_dict, "ambient": True}
        if precomputed:
            data = algorithm.calc_points(None, points=points, ambient=True)
            data[FV.WS] = xr.full_like(data[FV.WS], 17.0)
            options["point_results"] = data
        elif entry == "calc_points":
            options["points"] = points
        returned = getattr(algorithm, entry)(farm_results=None, **options)
    if ret_dict:
        assert set(returned) == {"speed"}
        returned = returned["speed"]
    np.testing.assert_allclose(
        returned[FV.WS].values[:, 0, 0],
        [17.0] * 3 if precomputed else [6.0, 8.0, 10.0],
    )


@pytest.mark.parametrize("outputs", [None, []])
@pytest.mark.parametrize("entry", ["calc_farm", "calc_points", "calc_outputs"])
def test_keyed_disabled_collection_stays_empty(algorithm, entry, outputs):
    # Key preservation never fabricates a result when collection is disabled.
    options = {"outputs": outputs, "ret_dict": True}
    if entry == "calc_points":
        options.update(farm_results=None, points=np.zeros((1, 3)), ambient=True)
    with foxes.Engine.new("numpy", verbosity=0):
        assert getattr(algorithm, entry)(**options) == {}


def test_keyed_output_errors_remain_explicit(algorithm):
    with foxes.Engine.new("numpy", verbosity=0):
        with pytest.raises(KeyError, match="missing_output"):
            algorithm.calc_farm(outputs="missing_output", ret_dict=True)


def test_output_only_farm_mapping_and_iterative_solver_history(algorithm):
    # Output-only collection and optional solver history have independent shapes.
    algorithm.farm.turbines[0].models = ["NREL5MW"]
    algo = foxes.algorithms.Iterative(
        algorithm.farm, algorithm.states, wake_models=[], verbosity=0
    )
    with foxes.Engine.new("numpy", verbosity=0):
        results, downwind = algo.calc_farm(ret_dict=True, ret_dwnd_order=True)
        returned = algo.calc_outputs(
            farm_results=results[FC.FARM_RESULTS],
            outputs="farm_results",
            ret_dict=True,
        )
    assert set(results) == set(returned) == {FC.FARM_RESULTS}
    xr.testing.assert_identical(returned[FC.FARM_RESULTS], results[FC.FARM_RESULTS])
    assert isinstance(downwind, xr.Dataset)


def test_multiple_planes_write_once(algorithm, tmp_path, monkeypatch):
    output = SlicesData(
        positions=[80.0, 100.0],
        variables=[FV.WS],
        grid_pars={"n_img_points": (2, 3)},
        to_file="planes.nc",
        out_dir=tmp_path,
    )
    writes = []
    original = output._write_data

    def write(data, verbosity, write_pars):
        writes.append(dict(data.sizes))
        assert verbosity == algorithm.verbosity
        original(data, verbosity, write_pars)

    monkeypatch.setattr(output, "_write_data", write)
    with foxes.Engine.new("numpy", chunk_size_points=2, verbosity=0):
        result = algorithm.calc_farm(outputs=output)
    assert len(writes) == 1
    np.testing.assert_array_equal(result["z"], [80.0, 100.0])
    with xr.open_dataset(tmp_path / "planes.nc") as written:
        xr.testing.assert_allclose(written, result)


def test_state_rose_without_farm_results(algorithm):
    output = StatesRosePlotOutput(point=(0.0, 0.0, 90.0), plot=False)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=1, verbosity=0
    ):
        result = algorithm.calc_outputs(outputs=output)
    assert float(result["frequency"].sum()) == pytest.approx(100.0)


def test_configured_yaml_output(algorithm):
    configuration = Dict(
        outputs=[
            Dict(
                output_type="SliceData",
                plane="xz",
                variables=[FV.WS],
                reduction="mean",
                grid_pars={"n_img_points": (2, 3)},
                result_labels="$slice",
            )
        ]
    )
    with foxes.Engine.new("numpy", chunk_size_points=2, verbosity=0):
        farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)
        outputs, storage = run_outputs(
            configuration,
            algorithm,
            farm_results=farm,
            with_engine=True,
            ret_results_storage=True,
            verbosity=0,
        )
    assert storage["$slice"] is outputs[0][1][0]
    np.testing.assert_allclose(storage["$slice"][FV.WS], 8.6)


def test_promoted_outputs_have_no_dataset_bound_aliases():
    names = (
        "FarmLayoutOutput",
        "FlowPlots2D",
        "SliceData",
        "SlicesData",
        "RosePlotOutput",
        "StatesRosePlotOutput",
        "WindRoseBinPlot",
    )
    for name in names:
        assert issubclass(getattr(foxes.models.outputs, name), foxes.core.Output)
        assert not hasattr(foxes, "output")
    output = FlowPlots2D(plane="yz", grid_pars={"n_img_points": (2, 3)})
    assert not any(
        isinstance(value, (np.ndarray, xr.Dataset)) for value in vars(output).values()
    )


def test_slice_and_flow_outputs_do_not_inherit_animation_behavior():
    assert issubclass(FlowAnimation, SliceData)
    assert issubclass(FlowPlots2D, SliceData)
    assert not issubclass(SliceData, FlowAnimation)
    assert not issubclass(FlowPlots2D, FlowAnimation)
    for output in (SliceData(), SlicesData([90.0]), FlowPlots2D(plot=False)):
        for attribute in ("base_name", "fps", "figsize", "plot_pars", "layout_pars"):
            assert not hasattr(output, attribute)
        assert not hasattr(output, "_render")


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("reduction", ["states", "mean"])
@pytest.mark.parametrize("plot", [False, True])
def test_named_flow_plot_uses_slice_data(
    algorithm, monkeypatch, capsys, plane, reduction, plot
):
    name = f"flow_plot_{plane}_{reduction}"
    output = algorithm.mbook.outputs[name]
    output.grid_pars = {"n_img_points": (2, 3)}
    output.plot = plot
    output.ret_data = True
    algorithm.verbosity = 2

    def forbidden(*args, **kwargs):
        raise AssertionError("Static flow plots must not initialize animation behavior")

    monkeypatch.setattr(FlowAnimation, "initialize", forbidden)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = algorithm.calc_farm(outputs=name)
    printed = capsys.readouterr().out
    assert output._grid_key == output.var("slice_points")
    assert output._result_key == output.var("slice_chunks")
    assert output._selection_key == output.var("slice_selection")
    assert output._grid_key in printed
    assert "animation" not in printed.lower()
    if plot:
        import matplotlib.pyplot as plt

        figures, data = result
        figures = [figures] if reduction == "mean" else figures
        try:
            assert len(figures) == (1 if reduction == "mean" else 3)
            for figure in figures:
                figure.canvas.draw()
        finally:
            for figure in figures:
                plt.close(figure)
    else:
        _, data, *_ = result
    assert data[FV.WS].dims == (
        (plane[1], plane[0]) if reduction == "mean" else (FC.STATE, plane[1], plane[0])
    )
    expected = 8.6 if reduction == "mean" else np.array([6.0, 8.0, 10.0])
    np.testing.assert_allclose(data[FV.WS].values[..., 0, 0], expected)


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("reduction", ["states", "mean"])
@pytest.mark.parametrize("data_format", ["xarray", "numpy"])
def test_flow_plot_data_reuses_reordered_precomputed_values(
    algorithm, monkeypatch, plane, reduction, data_format
):
    a, b = np.meshgrid([0.0, 20.0], [30.0, 40.0, 50.0], indexing="ij")
    points = np.full((6, 3), 90.0)
    points[:, "xyz".index(plane[0])] = a.ravel()
    points[:, "xyz".index(plane[1])] = b.ravel()
    order = [5, 1, 3, 0, 4, 2]
    values = np.array([6.0, 8.0, 10.0])[:, None] + np.arange(6)[None, :]
    precomputed = xr.Dataset(
        {
            FV.WS: ((FC.STATE, FC.POINT), values[:, order]),
            FV.WD: ((FC.STATE, FC.POINT), np.full((3, 6), 270.0)),
            FV.WEIGHT: ((FC.STATE,), [0.2, 0.3, 0.5]),
        },
        coords={
            FC.STATE: [10, 20, 30],
            FC.POINT: np.arange(6),
            **{
                variable: ((FC.POINT,), points[order, i])
                for i, variable in enumerate((FV.X, FV.Y, FV.H))
            },
        },
    )
    output = FlowPlots2D(
        plane=plane, reduction=reduction, data_format=data_format, plot=False
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Precomputed flow plots must use actual supplied data")

    monkeypatch.setattr(output, "get_calc_points", forbidden)
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        farm = algorithm.calc_farm(outputs="farm_results_all", finalize=False)
        monkeypatch.setattr(algorithm.states, "calculate", forbidden)
        _, data, *_ = algorithm.calc_outputs(
            outputs=output, farm_results=farm, point_results=precomputed
        )
    expected = values.reshape(3, 2, 3)
    if reduction == "mean":
        expected = np.einsum("s,sab->ab", [0.2, 0.3, 0.5], expected)
    if data_format == "xarray":
        actual = data[FV.WS].values
        expected = np.swapaxes(expected, -2, -1)
    else:
        actual = data[..., 0]
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("output_type", [SliceData, FlowPlots2D, SlicesData])
def test_mean_slice_worker_payload_has_no_state_frames(
    algorithm, monkeypatch, chunk_size, output_type
):
    kwargs = {"positions": [80.0, 100.0]} if output_type is SlicesData else {}
    if output_type is FlowPlots2D:
        kwargs["plot"] = False
    output = output_type(reduction="mean", grid_pars={"n_img_points": (2, 3)}, **kwargs)
    calculate = type(output).calculate
    payloads = []

    def capture(self, *args, **kwargs):
        result = calculate(self, *args, **kwargs)
        payload = result[self._result_key]
        assert payload["points"].shape[0] == 1
        assert payload["states"].shape == (1,)
        assert payload["order"].size == 0
        assert payload["farm"] == {}
        assert set(payload["values"]) == set(self.fields)
        count = payload["points"].shape[1]
        for values in payload["values"].values():
            assert values.shape == (1, count)
        payloads.append(payload)
        return result

    monkeypatch.setattr(type(output), "calculate", capture)
    with foxes.Engine.new(
        "numpy", chunk_size_states=chunk_size, chunk_size_points=2, verbosity=0
    ):
        algorithm.calc_farm(outputs=output)
    assert len(payloads) == (3 if chunk_size == 1 else 1)


def test_mean_slice_keeps_only_explicitly_requested_state_labels(algorithm):
    output = SliceData(
        reduction="mean",
        states_sel=[30, 10],
        ret_states=True,
        grid_pars={"n_img_points": (2, 3)},
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        data, states = algorithm.calc_farm(outputs=output)
    np.testing.assert_array_equal(states, [30, 10])
    np.testing.assert_allclose(data[FV.WS], 6.2)


@pytest.mark.parametrize("output_type", [SliceData, SlicesData, FlowPlots2D])
def test_slice_owns_flat_formatting_inputs(algorithm, monkeypatch, output_type):
    # Each family extends the slice lifecycle without copying formatting options.
    kwargs = {"positions": [80.0, 100.0]} if output_type is SlicesData else {}
    if output_type is FlowPlots2D:
        kwargs["plot"] = False
        kwargs["plot_pars"] = {"ax": object()}
    output = output_type(
        grid_pars={"n_img_points": (2, 3)},
        normalize={"x": 1000.0},
        normalize_v={FV.WS: 2.0},
        label_map={FV.WS: "speed"},
        vmin={FV.WS: 0.0},
        vmax={FV.WS: 20.0},
        **kwargs,
    )
    normalize, normalize_v = output.normalize, output.normalize_v
    label_map, write_pars = output.label_map, output.write_pars
    vmin, vmax = output.vmin, output.vmax
    format_payload = output.format_payload
    results_seen = []

    def format_selected(algo, payload, output_results):
        assert output.running
        assert not hasattr(output, "normalize")
        assert not hasattr(output, "normalize_v")
        assert not hasattr(output, "label_map")
        assert not hasattr(output, "write_pars")
        assert not hasattr(output, "vmin")
        assert not hasattr(output, "vmax")
        assert output_results[output.var("normalize")] is normalize
        assert output_results[output.var("normalize_v")] is normalize_v
        assert output_results[output.var("label_map")] is label_map
        assert output_results[output.var("write_pars")] is write_pars
        assert output_results[output.var("vmin")] is vmin
        assert output_results[output.var("vmax")] is vmax
        if isinstance(output, FlowPlots2D):
            assert not hasattr(output, "plot_options")
            assert (
                output_results[output.var("plot_options")]["ax"]
                is kwargs["plot_pars"]["ax"]
            )
        results_seen.append(output_results)
        return format_payload(algo, payload, output_results)

    monkeypatch.setattr(output, "format_payload", format_selected)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        results = algorithm.calc_farm(outputs=output, ret_dict=True)
    assert len(results_seen) == 1
    assert set(results_seen[0]) == set(results) == {output.result_name}
    assert output.normalize is normalize
    assert output.normalize_v is normalize_v
    assert output.label_map is label_map
    assert output.write_pars is write_pars
    assert output.vmin is vmin
    assert output.vmax is vmax


@pytest.mark.parametrize(
    "name,class_name",
    [
        ("farm_layout", "FarmLayoutOutput"),
        ("rose_plot", "RosePlotOutput"),
        ("states_rose_plot", "StatesRosePlotOutput"),
        ("wind_rose_bin_plot", "WindRoseBinPlot"),
        ("slice_data", "SliceData"),
        ("slices_data", "SlicesData"),
        ("flow_plot", "FlowPlots2D"),
    ],
)
def test_model_book_registers_adopted_outputs(name, class_name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Model-book construction must not load or calculate outputs"
        )

    cls = getattr(foxes.models.outputs, class_name)
    monkeypatch.setattr(cls, "initialize", forbidden)
    monkeypatch.setattr(cls, "calculate", forbidden)
    output = foxes.ModelBook().outputs[name]
    assert type(output) is cls
    assert isinstance(output, foxes.core.Output)
    assert output.name == name
    assert not output.initialized
    assert not any(
        isinstance(value, (np.ndarray, xr.Dataset)) for value in vars(output).values()
    )


@pytest.mark.parametrize("name", ["slice_data", "slices_data", "flow_plot"])
@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
@pytest.mark.parametrize("reduction", ["states", "mean"])
def test_model_book_slice_factories(name, plane, reduction):
    book = foxes.ModelBook()
    output = book.outputs[f"{name}_{plane}_{reduction}"]
    assert type(output) is type(book.outputs[name])
    assert output.plane == plane
    assert output.reduction == reduction
    assert not output.initialized
    if name == "slices_data":
        assert output.positions == (100.0,)


@pytest.mark.parametrize("suffix", ["invalid_mean", "xy_invalid"])
@pytest.mark.parametrize("name", ["slice_data", "slices_data", "flow_plot"])
def test_model_book_slice_factories_reject_invalid_choices(name, suffix):
    with pytest.raises(KeyError):
        foxes.ModelBook().outputs[f"{name}_{suffix}"]


def test_model_book_output_presets_are_independent():
    first, second = foxes.ModelBook(), foxes.ModelBook()
    first.outputs["slice_data"].grid_pars = {"n_img_points": (2, 3)}
    assert second.outputs["slice_data"].grid_pars is None
    assert first.outputs["slice_data_xz_mean"] is not first.outputs["slice_data"]
    assert first.outputs["states_rose_plot"].location == (0.0, 0.0, 100.0)
    assert first.outputs["wind_rose_bin_plot"].variable == FV.P
    assert first.outputs["wind_rose_bin_plot"].ws_bins == (0, 5, 10, 15, 20, 25)


def test_model_book_lists_adopted_output_factories(capsys):
    foxes.ModelBook().print_toc(subset=["outputs"])
    contents = capsys.readouterr().out
    for prefix in ("slice_data", "slices_data", "flow_plot"):
        assert f"{prefix}_<plane>_<reduction>" in contents
    assert "states_rose_plot: StatesRosePlotOutput" in contents
    assert "wind_rose_bin_plot: WindRoseBinPlot" in contents


def test_model_book_adopted_outputs_run_by_name(algorithm):
    # Configure the named instances before initialization, then retain list dispatch.
    output = algorithm.mbook.outputs["slice_data_xz_mean"]
    output.grid_pars = {"n_img_points": (2, 3)}
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        results = algorithm.calc_farm(
            outputs=["farm_results_all", "slice_data_xz_mean"]
        )
    np.testing.assert_allclose(results[output.result_name][FV.WS], 8.6)


def test_iterative_named_output_reuses_supplied_farm_results(algorithm, monkeypatch):
    algorithm.farm.turbines[0].models = ["NREL5MW"]
    algo = foxes.algorithms.Iterative(
        algorithm.farm,
        algorithm.states,
        wake_models=[],
        rotor_model="centre",
        mbook=algorithm.mbook,
        mod_cutin={"modify_ct": False},
        verbosity=0,
    )
    algo.mbook.outputs["selected_power"] = FarmResultsOutput(variables=[FV.P])
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        farm = algo.calc_farm(outputs="farm_results_all", finalize=False)
        farm[FV.P].values[:, 0] = [3.0, 7.0, 19.0]

        def forbidden(*args, **kwargs):
            raise AssertionError("Output-only runs must not recalculate supplied data")

        monkeypatch.setattr(algo.states, "calculate", forbidden)
        result = algo.calc_outputs(
            outputs="selected_power", farm_results=farm, states_sel=[10, 30]
        )
    np.testing.assert_allclose(result[FV.P].values[:, 0], [3.0, 19.0])
