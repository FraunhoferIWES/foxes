import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import PatchCollection, PathCollection

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.output.calc_points import PointCalculator
from foxes.output.farm_layout import FarmLayoutOutput
from foxes.output.farm_results_eval import FarmResultsEval
from foxes.output.farms_eval import WindFarmsEval
from foxes.output.flow_plots_2d.flow_plots import FlowPlots2D
from foxes.output.results_writer import ResultsWriter
from foxes.output.rose_plot import RosePlotOutput, WindRoseBinPlot
from foxes.output.state_turbine_table import StateTurbineTable
from foxes.utils.geom2d import ClosedPolygon


def _calc_farm_results():
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[400.0, 0.0],
        n_turbines=3,
        turbine_models=["NREL5MW"],
        H=90.0,
        verbosity=0,
    )

    states = foxes.input.states.SingleStateStates(
        ws=8.0,
        wd=270.0,
        ti=0.08,
        rho=1.225,
    )

    algo = foxes.algorithms.Downwind(
        farm=farm,
        states=states,
        wake_models=["Jensen_linear_k007"],
        verbosity=0,
    )

    with foxes.Engine.new("threads", verbosity=0):
        farm_results = algo.calc_farm()

    return algo, farm_results


def _calc_two_farm_results():
    farm = foxes.WindFarm()

    t0 = foxes.Turbine([0.0, 0.0], turbine_models=["NREL5MW"], H=90.0)
    t0.wind_farm_name = "west"
    farm.add_turbine(t0, verbosity=0)

    t1 = foxes.Turbine([400.0, 0.0], turbine_models=["NREL5MW"], H=90.0)
    t1.wind_farm_name = "east"
    farm.add_turbine(t1, verbosity=0)

    states = foxes.input.states.SingleStateStates(
        ws=8.0,
        wd=270.0,
        ti=0.08,
        rho=1.225,
    )

    algo = foxes.algorithms.Downwind(
        farm=farm,
        states=states,
        wake_models=["Jensen_linear_k007"],
        verbosity=0,
    )

    with foxes.Engine.new("threads", verbosity=0):
        farm_results = algo.calc_farm()

    return algo, farm_results


def test_results_writer_write_nc_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()
    out = ResultsWriter(farm_results=farm_results, out_dir=tmp_path)

    fname = "results_writer_smoke.nc"
    out.write_nc(fname, variables=[FV.P, FV.REWS], turbine_names=True, verbosity=0)

    fpath = tmp_path / fname
    assert fpath.is_file()

    ds = xr.open_dataset(fpath, engine=config.nc_engine)
    try:
        assert FV.P in ds.data_vars
        assert FV.REWS in ds.data_vars
        assert ds[FV.P].shape[0] == farm_results.sizes[FC.STATE]
        assert ds[FV.P].shape[1] == farm_results.sizes[FC.TURBINE]
    finally:
        ds.close()

    fpath.unlink()
    assert not fpath.exists()


def test_state_turbine_table_write_nc_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()
    out = StateTurbineTable(farm_results=farm_results, out_dir=tmp_path)

    fname = "state_turbine_table_smoke.nc"
    ds = out.get_dataset(
        variables=[FV.P, FV.REWS],
        name_map={FV.P: "power", FV.REWS: "rews"},
        to_file=fname,
    )

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert "power" in ds.data_vars
    assert "rews" in ds.data_vars

    check = xr.open_dataset(fpath, engine=config.nc_engine)
    try:
        assert "power" in check.data_vars
        assert check["power"].shape[0] == farm_results.sizes[FC.STATE]
        assert check["power"].shape[1] == farm_results.sizes[FC.TURBINE]
    finally:
        check.close()

    fpath.unlink()
    assert not fpath.exists()


def test_farm_results_eval_write_nc_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()
    out = FarmResultsEval(farm_results=farm_results, out_dir=tmp_path)

    fname = "farm_results_eval_smoke.nc"
    returned = out.write_nc(fname, nc_engine=config.nc_engine, verbosity=0)

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert FV.P in returned.data_vars

    ds = xr.open_dataset(fpath, engine=config.nc_engine)
    try:
        assert FV.P in ds.data_vars
        assert ds[FV.P].shape[0] == farm_results.sizes[FC.STATE]
        assert ds[FV.P].shape[1] == farm_results.sizes[FC.TURBINE]
    finally:
        ds.close()

    fpath.unlink()
    assert not fpath.exists()


def test_farm_results_eval_calc_yield_smoke():
    algo, farm_results = _calc_farm_results()
    out = FarmResultsEval(farm_results=farm_results, algo=algo)

    ambient_yield = out.calc_yield(annual=True, ambient=True)
    waked_yield = out.calc_yield(annual=True)

    assert list(ambient_yield.columns) == [FV.AMB_YLD]
    assert list(waked_yield.columns) == [FV.YLD]
    assert ambient_yield.shape[0] == farm_results.sizes[FC.TURBINE]
    assert waked_yield.shape[0] == farm_results.sizes[FC.TURBINE]
    assert np.all(np.isfinite(ambient_yield[FV.AMB_YLD].to_numpy()))
    assert np.all(np.isfinite(waked_yield[FV.YLD].to_numpy()))


def test_farm_results_eval_calc_farm_mean():
    rews = np.array([[1.0, 10.0], [3.0, 14.0]])
    weight_cases = [
        ((FC.STATE,), np.array([0.25, 0.75])),
        (
            (FC.STATE, FC.TURBINE),
            np.array([[0.25, 0.75], [0.75, 0.25]]),
        ),
    ]

    for weight_dims, weights in weight_cases:
        farm_results = xr.Dataset(
            data_vars={
                FV.REWS: ((FC.STATE, FC.TURBINE), rews),
                FV.WEIGHT: (weight_dims, weights),
            },
            coords={FC.STATE: np.arange(2), FC.TURBINE: np.arange(2)},
        )
        state_weights = weights[:, None] if weights.ndim == 1 else weights
        expected = np.mean(np.sum(rews * state_weights, axis=0))

        result = FarmResultsEval(farm_results).calc_farm_mean([FV.REWS])

        assert np.isclose(result[FV.REWS], expected)

        evaluator = FarmResultsEval(farm_results)
        turbine_mean = evaluator.reduce_turbines({FV.REWS: "weights"})
        turbine_sum = evaluator.reduce_turbines({FV.REWS: "weights_sum"})
        expected_by_state = np.sum(rews * state_weights, axis=1)

        assert np.allclose(turbine_mean[FV.REWS], expected_by_state / rews.shape[1])
        assert np.allclose(turbine_sum[FV.REWS], expected_by_state)

        farm_sum = evaluator.reduce_all(
            states_op={FV.REWS: "weights"},
            turbines_op={FV.REWS: "weights_sum"},
        )

        assert np.isclose(farm_sum[FV.REWS], expected * rews.shape[1])


@pytest.mark.parametrize(
    ("ambient", "expected"),
    [
        (False, 0.45),
        (True, 0.7),
    ],
)
def test_farm_results_eval_calc_farm_capacity_factor(ambient, expected):
    farm_results = xr.Dataset(
        data_vars={
            FV.P: ((FC.STATE, FC.TURBINE), [[20.0, 40.0], [60.0, 100.0]]),
            FV.AMB_P: ((FC.STATE, FC.TURBINE), [[40.0, 80.0], [80.0, 160.0]]),
            FV.CAP: ((FC.TURBINE,), [100.0, 200.0]),
            FV.WEIGHT: ((FC.STATE,), [0.25, 0.75]),
        },
        coords={FC.STATE: np.arange(2), FC.TURBINE: np.arange(2)},
    )

    result = FarmResultsEval(farm_results).calc_farm_capacity_factor(ambient=ambient)

    assert np.isclose(result, expected)


def test_farm_results_eval_calc_farm_capacity_factor_with_state_capacity():
    farm_results = xr.Dataset(
        data_vars={
            FV.P: ((FC.STATE, FC.TURBINE), [[20.0, 40.0], [60.0, 100.0]]),
            FV.CAP: ((FC.STATE, FC.TURBINE), [[100.0, 200.0], [100.0, 200.0]]),
            FV.WEIGHT: ((FC.STATE,), [0.25, 0.75]),
        },
        coords={FC.STATE: np.arange(2), FC.TURBINE: np.arange(2)},
    )

    result = FarmResultsEval(farm_results).calc_farm_capacity_factor()

    assert np.isclose(result, 0.45)


def test_farm_results_eval_calc_farm_capacity_factor_rejects_zero_capacity():
    farm_results = xr.Dataset(
        data_vars={
            FV.P: ((FC.STATE, FC.TURBINE), [[10.0]]),
            FV.CAP: ((FC.TURBINE,), [0.0]),
            FV.WEIGHT: ((FC.STATE,), [1.0]),
        },
        coords={FC.STATE: np.arange(1), FC.TURBINE: np.arange(1)},
    )

    with pytest.raises(ValueError, match="Farm capacity must be positive"):
        FarmResultsEval(farm_results).calc_farm_capacity_factor()


def test_farm_results_eval_ignores_only_zero_weight_nans():
    rews = np.array([[1.0, np.nan], [3.0, 4.0]])
    weights = np.array([[0.5, 0.0], [0.5, 1.0]])
    farm_results = xr.Dataset(
        data_vars={
            FV.REWS: ((FC.STATE, FC.TURBINE), rews),
            FV.WEIGHT: ((FC.STATE, FC.TURBINE), weights),
        },
        coords={FC.STATE: np.arange(2), FC.TURBINE: np.arange(2)},
    )

    evaluator = FarmResultsEval(farm_results)
    state_mean = evaluator.calc_states_mean(FV.REWS)
    turbine_mean = evaluator.calc_turbine_mean([FV.REWS])

    np.testing.assert_allclose(state_mean[FV.REWS], [2.0, 4.0])
    np.testing.assert_allclose(turbine_mean[FV.REWS], [1.0, 3.5])
    assert np.isclose(evaluator.calc_farm_mean([FV.REWS])[FV.REWS], 3.0)

    farm_results[FV.WEIGHT].data[0, 1] = 0.5
    with pytest.raises(AssertionError, match="nonzero weights"):
        FarmResultsEval(farm_results).calc_states_mean(FV.REWS)


def test_point_calculator_write_nc_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()
    out = PointCalculator(algo=algo, farm_results=farm_results, out_dir=tmp_path)

    points = np.array([[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]], dtype=float)
    fname = "point_calculator_smoke.nc"
    pres = out.calculate(
        points=points,
        to_file=fname,
        write_vars=[FV.WS],
        write_pars={"verbosity": 0},
    )

    assert FV.WS in pres.data_vars
    fpath = tmp_path / fname
    assert fpath.is_file()

    ds = xr.open_dataset(fpath, engine=config.nc_engine)
    try:
        assert FV.WS in ds.data_vars
        assert "x" in ds.data_vars
        assert "y" in ds.data_vars
        assert "z" in ds.data_vars
        assert ds[FV.WS].shape[0] == farm_results.sizes[FC.STATE]
        assert ds[FV.WS].shape[1] == points.shape[0]
    finally:
        ds.close()

    fpath.unlink()
    assert not fpath.exists()


def test_farm_layout_output_write_plot_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()
    out = FarmLayoutOutput(farm=algo.farm, farm_results=farm_results, out_dir=tmp_path)

    fname = "farm_layout_smoke.png"
    out.write_plot(file_name=fname)

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_farm_layout_output_figure_accepts_default_boundary_args():
    farm = foxes.WindFarm(boundary=foxes.utils.geom2d.Circle([0.0, 0.0], 1000.0))
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[400.0, 0.0],
        n_turbines=1,
        turbine_models=["NREL5MW"],
        H=90.0,
        verbosity=0,
    )

    ax = FarmLayoutOutput(farm=farm).get_figure()
    plt.close(ax.get_figure())


def test_farm_layout_output_auto_figsize_and_colors():
    farm = foxes.WindFarm()
    for xy in ([0.0, 0.0], [1000.0, 100.0]):
        farm.add_turbine(foxes.Turbine(xy, turbine_models=[], D=100.0, H=90.0))

    ax = FarmLayoutOutput(farm=farm).get_figure(
        c=np.array(["red", "orange"]),
        legend_labels={
            "orange": "Valid turbine",
            "red": "Constraint violation",
        },
    )
    width, height = ax.get_figure().get_size_inches()
    colors = ax.collections[0].get_facecolors()
    legend = ax.get_legend()
    legend_labels = [text.get_text() for text in legend.get_texts()]
    legend_location = legend._loc
    plt.close(ax.get_figure())

    assert width > height
    np.testing.assert_allclose(colors[0, :3], [1.0, 0.0, 0.0])
    assert legend_labels == ["Valid turbine", "Constraint violation"]
    assert legend_location == legend.codes["upper left"]

    ax = FarmLayoutOutput(farm=farm).get_figure(figsize=(3.0, 4.0))
    np.testing.assert_allclose(ax.get_figure().get_size_inches(), [3.0, 4.0])
    plt.close(ax.get_figure())


def test_farm_layout_output_true_turbine_radii_are_opt_in():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[], D=100.0, H=90.0))
    farm.add_turbine(foxes.Turbine([1000.0, 100.0], turbine_models=[], D=200.0, H=90.0))

    colors = np.array(["red", "orange"])
    default_ax = FarmLayoutOutput(farm=farm).get_figure(annotate=0, c=colors)
    assert isinstance(default_ax.collections[0], PathCollection)

    radius_ax = FarmLayoutOutput(farm=farm).get_figure(
        annotate=0,
        true_turbine_radii=True,
        c=colors,
    )
    collection = radius_ax.collections[0]
    widths = [path.get_extents().width for path in collection.get_paths()]
    np.testing.assert_allclose(
        collection.get_facecolors(), default_ax.collections[0].get_facecolors()
    )
    np.testing.assert_allclose(collection.get_edgecolors(), collection.get_facecolors())
    plt.close(default_ax.get_figure())
    plt.close(radius_ax.get_figure())

    assert isinstance(collection, PatchCollection)
    np.testing.assert_allclose(widths, [100.0, 200.0])


def test_farm_layout_output_true_radii_auto_figsize_has_visible_rotors():
    farm = foxes.WindFarm()
    for xy in ([0.0, 0.0], [20000.0, 100.0]):
        farm.add_turbine(foxes.Turbine(xy, turbine_models=[], D=100.0, H=90.0))
    output = FarmLayoutOutput(farm=farm)

    default_ax = output.get_figure(annotate=0)
    radius_ax = output.get_figure(annotate=0, true_turbine_radii=True)
    radius_ax.get_figure().canvas.draw()
    p0, p1 = radius_ax.transData.transform([[0.0, 0.0], [100.0, 0.0]])

    assert (
        radius_ax.get_figure().get_figwidth() > default_ax.get_figure().get_figwidth()
    )
    assert p1[0] - p0[0] >= 4.0
    plt.close(default_ax.get_figure())
    plt.close(radius_ax.get_figure())

    explicit_ax = output.get_figure(
        annotate=0,
        figsize=(3.0, 4.0),
        true_turbine_radii=True,
    )
    np.testing.assert_allclose(explicit_ax.get_figure().get_size_inches(), [3.0, 4.0])
    plt.close(explicit_ax.get_figure())


def test_farm_layout_output_true_turbine_radii_support_color_by():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[], D=100.0, H=90.0))
    farm.add_turbine(foxes.Turbine([1000.0, 100.0], turbine_models=[], D=100.0, H=90.0))
    farm_results = xr.Dataset({"score": ((FC.TURBINE,), [1.0, 2.0])})
    output = FarmLayoutOutput(farm=farm, farm_results=farm_results)
    plot_pars = {
        "color_by": "score",
        "cmap": "plasma",
        "vmin": 0.0,
        "vmax": 3.0,
        "alpha": 0.6,
    }

    scatter_ax = output.get_figure(annotate=0, **plot_pars)
    radius_ax = output.get_figure(
        annotate=0,
        true_turbine_radii=True,
        **plot_pars,
    )
    scatter_ax.get_figure().canvas.draw()
    radius_ax.get_figure().canvas.draw()
    scatter = scatter_ax.collections[0]
    circles = radius_ax.collections[0]

    np.testing.assert_allclose(circles.get_array(), scatter.get_array())
    np.testing.assert_allclose(circles.get_clim(), scatter.get_clim())
    np.testing.assert_allclose(circles.get_facecolors(), scatter.get_facecolors())
    np.testing.assert_allclose(circles.get_edgecolors(), scatter.get_edgecolors())
    plt.close(scatter_ax.get_figure())
    plt.close(radius_ax.get_figure())


def test_farm_layout_output_true_turbine_radii_require_diameters():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[], H=90.0))

    ax = FarmLayoutOutput(farm=farm, D=100.0).get_figure(normalize_D=True)
    plt.close(ax.get_figure())

    with pytest.raises(ValueError, match="finite positive rotor diameters"):
        FarmLayoutOutput(farm=farm).get_figure(true_turbine_radii=True)


def test_layout2d_figure_write_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()
    out = FarmLayoutOutput(
        farm=algo.farm,
        farm_results=farm_results,
        from_results=True,
        results_state=0,
        out_dir=tmp_path,
    )

    fname = "layout2d_smoke.png"
    out.write_plot(file_name=fname)

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_rose_plot_output_write_figure_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()
    out = RosePlotOutput(farm_results=farm_results, out_dir=tmp_path)

    fname = "rose_plot_smoke.png"
    out.write_figure(
        file_name=fname,
        wd_sectors=12,
        ws_var=FV.AMB_REWS,
        ws_bins=[0.0, 4.0, 8.0, 12.0, 16.0],
        turbine=0,
    )

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_wind_rose_bin_plot_write_figure_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()
    out = WindRoseBinPlot(farm_results=farm_results, out_dir=tmp_path)

    fname = "wind_rose_bin_smoke.png"
    out.write_figure(
        file_name=fname,
        variable=FV.P,
        ws_bins=[0.0, 4.0, 8.0, 12.0, 16.0],
        wd_sectors=12,
        ws_var=FV.AMB_REWS,
        wd_var=FV.AMB_WD,
        turbine=0,
    )

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_wind_farms_eval_area_mapping_plot_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_two_farm_results()
    out = WindFarmsEval(farm=algo.farm, farm_results=farm_results, out_dir=tmp_path)

    areas = {
        "west": ClosedPolygon(
            np.array(
                [[-200.0, -200.0], [200.0, -200.0], [200.0, 200.0], [-200.0, 200.0]]
            )
        ),
        "east": ClosedPolygon(
            np.array([[200.0, -200.0], [600.0, -200.0], [600.0, 200.0], [200.0, 200.0]])
        ),
    }

    fname = "wind_farms_area_mapping_smoke.png"
    out.write_area_mapping_plot(plot_file=fname, areas=areas, verbosity=0)

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_flow_plots2d_slice_data_write_nc_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()
    out = FlowPlots2D(algo=algo, farm_results=farm_results, out_dir=tmp_path)

    fname = "flow_field_xy_smoke.nc"
    params, data, _ = out.get_mean_data_xy(
        var=FV.WS,
        data_format="xarray",
        n_img_points=(12, 10),
        z=90.0,
        to_file=fname,
        verbosity=0,
    )

    assert params["var"] == FV.WS
    assert FV.WS in data.data_vars

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    ds = xr.open_dataset(fpath, engine=config.nc_engine)
    try:
        assert FV.WS in ds.data_vars
    finally:
        ds.close()

    fpath.unlink()
    assert not fpath.exists()


def test_flow_plots2d_figure_write_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()
    out = FlowPlots2D(algo=algo, farm_results=farm_results, out_dir=tmp_path)

    mean_data_xy = out.get_mean_data_xy(
        var=FV.WS,
        data_format="numpy",
        n_img_points=(12, 10),
        z=90.0,
        verbosity=0,
    )

    fig = out.get_mean_fig_xy(mean_data_xy)
    fpath = tmp_path / "flow_field_xy_smoke.png"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)

    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()
