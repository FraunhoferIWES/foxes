"""Noninteractive plotting, encoding and figure cleanup for output models."""

from threading import get_ident

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import foxes
import foxes.variables as FV
from foxes.models.outputs import (
    FarmLayoutOutput,
    FlowAnimation,
    FlowPlots2D,
    FlowTurbineAnimation,
    RosePlotOutput,
    SliceData,
    StatesRosePlotOutput,
    WindRoseBinPlot,
)


@pytest.fixture
def algorithm():
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WS: [6.0, 8.0, 10.0], FV.WEIGHT: [0.2, 0.3, 0.5]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine([0.0, 0.0], H=90.0, D=100.0, turbine_models=["NREL5MW"]),
        verbosity=0,
    )
    return foxes.algorithms.Downwind(
        farm, states, wake_models=[], rotor_model="centre", verbosity=0
    )


@pytest.fixture(autouse=True)
def display(monkeypatch):
    baseline = set(plt.get_fignums())
    main_thread = get_ident()
    calls = []

    def show(*, block):
        assert block is True
        assert get_ident() == main_thread
        assert plt.get_fignums()
        calls.append(tuple(plt.get_fignums()))

    monkeypatch.setattr(plt, "show", show)
    yield calls
    for number in set(plt.get_fignums()) - baseline:
        plt.close(number)


def _output(kind, **kwargs):
    if kind == "layout":
        return FarmLayoutOutput(**kwargs)
    if kind == "layout_writer":
        return FarmLayoutOutput(
            operation="write_plot", parameters={"file_name": "layout.png"}, **kwargs
        )
    if kind == "flow":
        return FlowPlots2D(grid_pars={"n_img_points": (2, 3)}, **kwargs)
    if kind == "rose":
        return RosePlotOutput(**kwargs)
    if kind == "state_rose":
        return StatesRosePlotOutput(point=(0.0, 0.0, 90.0), **kwargs)
    if kind == "binned_rose":
        return WindRoseBinPlot(variable=FV.P, ws_bins=(0, 5, 10, 15), **kwargs)
    animation = {"animation": FlowAnimation, "turbine_animation": FlowTurbineAnimation}[
        kind
    ]
    return animation(
        base_name="flow.gif", grid_pars={"n_img_points": (2, 3)}, fps=2.5, **kwargs
    )


PLOTS = (
    "layout",
    "layout_writer",
    "flow",
    "rose",
    "state_rose",
    "binned_rose",
    "animation",
    "turbine_animation",
)


@pytest.mark.parametrize("kind", PLOTS)
def test_direct_plot_constructors_do_not_show_by_default(
    algorithm, tmp_path, display, kind
):
    output = _output(kind, out_dir=tmp_path)
    assert output.show is False
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        algorithm.calc_farm(outputs=output)
    assert not display


@pytest.mark.parametrize("kind", PLOTS)
def test_threaded_plotting_never_displays(algorithm, tmp_path, display, kind):
    output = _output(kind, show=False, out_dir=tmp_path)
    with foxes.Engine.new(
        "threads", n_procs=2, chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        result = algorithm.calc_farm(outputs=output)
    assert not display
    if kind in ("animation", "turbine_animation"):
        with Image.open(tmp_path / "flow_WS.gif") as image:
            assert image.n_frames == 3
    elif kind == "flow":
        assert result.axes
    elif kind in ("rose", "state_rose", "binned_rose"):
        axes, data = result
        assert axes.figure is not None
        assert data.data_vars


@pytest.mark.parametrize("kind", PLOTS)
def test_nofig_prevents_plotting_and_file_creation(algorithm, tmp_path, display, kind):
    output = _output(kind, show=False, nofig=True, out_dir=tmp_path)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=output)
    assert not display
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["flow", "rose", "state_rose", "binned_rose"])
def test_numerical_plot_mode_never_shows(algorithm, tmp_path, display, kind):
    output = _output(kind, show=False, plot=False, out_dir=tmp_path)
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        result = algorithm.calc_farm(outputs=output)
    assert result is not None
    assert not display


@pytest.mark.parametrize(
    "kind",
    ["layout_writer", "flow", "rose", "binned_rose", "animation", "turbine_animation"],
)
def test_files_are_saved_without_display_and_transient_figures_are_closed(
    algorithm, tmp_path, display, kind
):
    options = (
        {}
        if kind in ("layout_writer", "animation", "turbine_animation")
        else {"to_file": "plot.png"}
    )
    output = _output(kind, show=False, out_dir=tmp_path, **options)
    original = set(plt.get_fignums())

    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=output)
    assert list(tmp_path.iterdir())
    assert not display
    if kind in ("layout_writer", "animation", "turbine_animation"):
        assert set(plt.get_fignums()) == original


@pytest.mark.parametrize(
    "name",
    [
        "farm_layout",
        "flow_plot",
        "flow_plot_xy_mean",
        "flow_plot_xz_states",
        "rose_plot",
        "states_rose_plot",
        "wind_rose_bin_plot",
        "create_gif",
        "create_mp4",
        "create_turbine_gif",
        "create_turbine_mp4",
        "chunk_gif_0",
        "chunk_mp4_1",
    ],
)
def test_model_book_test_presets_disable_display(name, display):
    output = foxes.ModelBook().outputs[name]
    assert output.show is False
    assert not display


@pytest.mark.parametrize(
    "name",
    [
        "farm_results",
        "point_results",
        "slice_data_xy_mean",
        "slices_data_xz_states",
    ],
)
def test_model_book_numerical_presets_do_not_enable_display(name):
    assert foxes.ModelBook().outputs[name].show is False


@pytest.mark.parametrize(
    "kind", ["layout", "flow", "rose", "state_rose", "binned_rose"]
)
def test_model_book_display_can_be_disabled(algorithm, tmp_path, display, kind):
    name = {
        "layout": "farm_layout",
        "flow": "flow_plot_xy_mean",
        "rose": "rose_plot",
        "state_rose": "states_rose_plot",
        "binned_rose": "wind_rose_bin_plot",
    }[kind]
    output = algorithm.mbook.outputs[name]
    output.show = False
    output.out_dir = tmp_path
    if kind == "flow":
        output.grid_pars = {"n_img_points": (2, 3)}
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=name)
    assert not display


@pytest.mark.parametrize(
    "name",
    [
        "farm_layout",
        "flow_plot_xy_mean",
        "rose_plot",
        "states_rose_plot",
        "wind_rose_bin_plot",
        "create_gif",
        "create_turbine_gif",
        "chunk_gif_0",
    ],
)
def test_model_book_named_execution_never_displays(algorithm, tmp_path, display, name):
    output = algorithm.mbook.outputs[name]
    output.show = False
    output.out_dir = tmp_path
    if isinstance(output, (FlowPlots2D, FlowAnimation)):
        output.grid_pars = {"n_img_points": (2, 3)}
    with foxes.Engine.new(
        "numpy", chunk_size_states=2, chunk_size_points=2, verbosity=0
    ):
        algorithm.calc_farm(outputs=name)
    assert not display


@pytest.mark.parametrize("animation_type", [FlowAnimation, FlowTurbineAnimation])
def test_animation_uses_configured_playback_rate_without_display(
    algorithm, tmp_path, monkeypatch, display, animation_type
):
    from foxes.models.outputs import Animator

    original = Animator.animate
    intervals = []

    def create(self, **kwargs):
        animation = original(self, **kwargs)
        intervals.append(animation.event_source.interval)
        return animation

    monkeypatch.setattr(Animator, "animate", create)
    output = animation_type(
        show=False,
        fps=2.5,
        base_name="flow.gif",
        out_dir=tmp_path,
        grid_pars={"n_img_points": (2, 3)},
    )
    before = set(plt.get_fignums())
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=output)
    assert intervals == [400]
    assert (tmp_path / "flow_WS.gif").is_file()
    assert not display
    assert set(plt.get_fignums()) == before


def test_mean_plot_returns_values_without_display(algorithm, tmp_path, display):
    output = FlowPlots2D(
        show=False, ret_data=True, out_dir=tmp_path, grid_pars={"n_img_points": (2, 3)}
    )
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        figure, data = algorithm.calc_farm(outputs=output)
    assert figure.axes and not display
    np.testing.assert_allclose(data[FV.WS], 8.6)


def test_numerical_slices_do_not_display(algorithm, tmp_path, display):
    output = SliceData(show=False, out_dir=tmp_path, grid_pars={"n_img_points": (2, 3)})
    with foxes.Engine.new("numpy", chunk_size_states=2, verbosity=0):
        algorithm.calc_farm(outputs=output)
    assert not display
