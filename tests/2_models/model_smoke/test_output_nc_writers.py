from _output_contract_helpers import run_output
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
from foxes.output.farms_eval import WindFarmsEval
from foxes.output.results_writer import ResultsWriter
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
        farm_results = algo.calc_farm(outputs="farm_results_all")

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
        farm_results = algo.calc_farm(outputs="farm_results_all")

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

    fname = "farm_results_eval_smoke.nc"
    returned = run_output(
        foxes.models.outputs.FarmResultsEval(
            out_dir=tmp_path,
            operation="write_nc",
            parameters=dict(fname=fname, nc_engine=config.nc_engine, verbosity=0),
        ),
        farm_results=farm_results,
    )

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

    ambient_yield = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_yield", parameters=dict(annual=True, ambient=True)
        ),
        farm_results=farm_results,
        algo=algo,
    )
    waked_yield = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_yield", parameters=dict(annual=True)
        ),
        farm_results=farm_results,
        algo=algo,
    )

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

        result = run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="calc_farm_mean", parameters=dict(vars=[FV.REWS])
            ),
            farm_results=farm_results,
        )

        assert np.isclose(result[FV.REWS], expected)

        turbine_mean = run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="reduce_turbines",
                parameters=dict(vars_op={FV.REWS: "weights"}),
            ),
            farm_results=farm_results,
        )
        turbine_sum = run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="reduce_turbines",
                parameters=dict(vars_op={FV.REWS: "weights_sum"}),
            ),
            farm_results=farm_results,
        )
        expected_by_state = np.sum(rews * state_weights, axis=1)

        assert np.allclose(turbine_mean[FV.REWS], expected_by_state / rews.shape[1])
        assert np.allclose(turbine_sum[FV.REWS], expected_by_state)

        farm_sum = run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="reduce_all",
                parameters=dict(
                    states_op={FV.REWS: "weights"}, turbines_op={FV.REWS: "weights_sum"}
                ),
            ),
            farm_results=farm_results,
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

    result = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_farm_capacity_factor", parameters=dict(ambient=ambient)
        ),
        farm_results=farm_results,
    )

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

    result = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_farm_capacity_factor", parameters=dict()
        ),
        farm_results=farm_results,
    )

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
        run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="calc_farm_capacity_factor", parameters=dict()
            ),
            farm_results=farm_results,
        )


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

    state_mean = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_states_mean", parameters=dict(vars=FV.REWS)
        ),
        farm_results=farm_results,
    )
    turbine_mean = run_output(
        foxes.models.outputs.FarmResultsEval(
            operation="calc_turbine_mean", parameters=dict(vars=[FV.REWS])
        ),
        farm_results=farm_results,
    )

    np.testing.assert_allclose(state_mean[FV.REWS], [2.0, 4.0])
    np.testing.assert_allclose(turbine_mean[FV.REWS], [1.0, 3.5])
    assert np.isclose(
        run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="calc_farm_mean", parameters=dict(vars=[FV.REWS])
            ),
            farm_results=farm_results,
        )[FV.REWS],
        3.0,
    )

    farm_results[FV.WEIGHT].data[0, 1] = 0.5
    with pytest.raises(AssertionError, match="nonzero weights"):
        run_output(
            foxes.models.outputs.FarmResultsEval(
                operation="calc_states_mean", parameters=dict(vars=FV.REWS)
            ),
            farm_results=farm_results,
        )


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

    fname = "farm_layout_smoke.png"
    run_output(
        foxes.models.outputs.FarmLayoutOutput(
            out_dir=tmp_path, operation="write_plot", parameters=dict(file_name=fname)
        ),
        farm_results=farm_results,
        farm=algo.farm,
    )

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

    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict()
        ),
        farm=farm,
    )
    plt.close(ax.get_figure())


def test_farm_layout_output_auto_figsize_and_colors():
    farm = foxes.WindFarm()
    for xy in ([0.0, 0.0], [1000.0, 100.0]):
        farm.add_turbine(foxes.Turbine(xy, turbine_models=[], D=100.0, H=90.0))

    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure",
            parameters=dict(
                c=np.array(["red", "orange"]),
                legend_labels={
                    "orange": "Valid turbine",
                    "red": "Constraint violation",
                },
            ),
        ),
        farm=farm,
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

    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(figsize=(3.0, 4.0))
        ),
        farm=farm,
    )
    np.testing.assert_allclose(ax.get_figure().get_size_inches(), [3.0, 4.0])
    plt.close(ax.get_figure())


@pytest.mark.parametrize(
    ("reference_names", "reference_label"),
    [
        (["reference point"], "reference point"),
        (["reference point 1", "reference point 2"], "reference points"),
    ],
)
def test_farm_layout_output_reference_point_defaults(reference_names, reference_label):
    farm = foxes.WindFarm(name="farm plus ref point")
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[]))
    for index, name in enumerate(reference_names, start=1):
        farm.add_turbine(
            foxes.Turbine(
                [1000.0 * index, 0.0],
                turbine_models=[],
                name=name,
            )
        )

    fig, ax = plt.subplots()
    ax.plot([0.0, 2000.0], [100.0, 100.0], label="CFD grid points")
    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(fig=fig, ax=ax, annotate=0)
        ),
        farm=farm,
    )
    legend = ax.get_legend()

    assert ax.get_title() == ""
    assert [text.get_text() for text in legend.get_texts()] == [
        "CFD grid points",
        "farm turbines",
        reference_label,
    ]
    ax.figure.canvas.draw()
    colors = ax.collections[0].get_facecolors()[:, :3]
    np.testing.assert_allclose(
        colors,
        [[1.0, 0.64705882, 0.0]] + [[1.0, 0.0, 0.0]] * len(reference_names),
    )
    plt.close(ax.get_figure())


def test_farm_layout_output_reference_point_custom_color_is_preserved():
    farm = foxes.WindFarm(name="farm plus ref point")
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[]))
    farm.add_turbine(
        foxes.Turbine(
            [1000.0, 0.0],
            turbine_models=[],
            name="reference point",
        )
    )

    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure",
            parameters=dict(annotate=0, title="custom title", c="green"),
        ),
        farm=farm,
    )

    assert ax.get_title() == "custom title"
    assert ax.get_legend() is None
    plt.close(ax.get_figure())


def test_farm_layout_output_true_turbine_radii_are_opt_in():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[], D=100.0, H=90.0))
    farm.add_turbine(foxes.Turbine([1000.0, 100.0], turbine_models=[], D=200.0, H=90.0))

    colors = np.array(["red", "orange"])
    default_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(annotate=0, c=colors)
        ),
        farm=farm,
    )
    assert isinstance(default_ax.collections[0], PathCollection)

    radius_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure",
            parameters=dict(annotate=0, true_turbine_radii=True, c=colors),
        ),
        farm=farm,
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

    default_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(annotate=0)
        ),
        farm=farm,
    )
    radius_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(annotate=0, true_turbine_radii=True)
        ),
        farm=farm,
    )
    radius_ax.get_figure().canvas.draw()
    p0, p1 = radius_ax.transData.transform([[0.0, 0.0], [100.0, 0.0]])

    assert (
        radius_ax.get_figure().get_figwidth() > default_ax.get_figure().get_figwidth()
    )
    assert p1[0] - p0[0] >= 4.0
    plt.close(default_ax.get_figure())
    plt.close(radius_ax.get_figure())

    explicit_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure",
            parameters=dict(annotate=0, figsize=(3.0, 4.0), true_turbine_radii=True),
        ),
        farm=farm,
    )
    np.testing.assert_allclose(explicit_ax.get_figure().get_size_inches(), [3.0, 4.0])
    plt.close(explicit_ax.get_figure())


def test_farm_layout_output_true_turbine_radii_support_color_by():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=[], D=100.0, H=90.0))
    farm.add_turbine(foxes.Turbine([1000.0, 100.0], turbine_models=[], D=100.0, H=90.0))
    farm_results = xr.Dataset({"score": ((FC.TURBINE,), [1.0, 2.0])})

    plot_pars = {
        "color_by": "score",
        "cmap": "plasma",
        "vmin": 0.0,
        "vmax": 3.0,
        "alpha": 0.6,
    }

    scatter_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure", parameters=dict(annotate=0, **plot_pars)
        ),
        farm_results=farm_results,
        farm=farm,
    )
    radius_ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            operation="get_figure",
            parameters=dict(annotate=0, true_turbine_radii=True, **plot_pars),
        ),
        farm_results=farm_results,
        farm=farm,
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

    ax = run_output(
        foxes.models.outputs.FarmLayoutOutput(
            D=100.0, operation="get_figure", parameters=dict(normalize_D=True)
        ),
        farm=farm,
    )
    plt.close(ax.get_figure())

    with pytest.raises(ValueError, match="finite positive rotor diameters"):
        run_output(
            foxes.models.outputs.FarmLayoutOutput(
                operation="get_figure", parameters=dict(true_turbine_radii=True)
            ),
            farm=farm,
        )


def test_layout2d_figure_write_smoke_and_cleanup(tmp_path):
    algo, farm_results = _calc_farm_results()

    fname = "layout2d_smoke.png"
    run_output(
        foxes.models.outputs.FarmLayoutOutput(
            from_results=True,
            results_state=0,
            out_dir=tmp_path,
            operation="write_plot",
            parameters=dict(file_name=fname),
        ),
        farm_results=farm_results,
        farm=algo.farm,
    )

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_rose_plot_output_write_figure_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()

    fname = "rose_plot_smoke.png"
    ax, _ = run_output(
        foxes.models.outputs.RosePlotOutput(
            out_dir=tmp_path,
            wd_sectors=12,
            ws_var=FV.AMB_REWS,
            ws_bins=[0.0, 4.0, 8.0, 12.0, 16.0],
            turbine=0,
            plot=True,
            to_file=fname,
        ),
        farm_results=farm_results,
    )
    plt.close(ax.get_figure())

    fpath = tmp_path / fname
    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()


def test_wind_rose_bin_plot_write_figure_smoke_and_cleanup(tmp_path):
    _, farm_results = _calc_farm_results()

    fname = "wind_rose_bin_smoke.png"
    ax, _ = run_output(
        foxes.models.outputs.WindRoseBinPlot(
            out_dir=tmp_path,
            variable=FV.P,
            ws_bins=[0.0, 4.0, 8.0, 12.0, 16.0],
            wd_sectors=12,
            ws_var=FV.AMB_REWS,
            wd_var=FV.AMB_WD,
            turbine=0,
            plot=True,
            to_file=fname,
        ),
        farm_results=farm_results,
    )
    plt.close(ax.get_figure())

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

    fname = "flow_field_xy_smoke.nc"
    params, data, _ = run_output(
        foxes.models.outputs.FlowPlots2D(
            out_dir=tmp_path,
            plane="xy",
            reduction="mean",
            plot=False,
            data_format="xarray",
            var=FV.WS,
            to_file=fname,
            grid_pars=dict(n_img_points=(12, 10), z=90.0),
        ),
        farm_results=farm_results,
        algo=algo,
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

    fig = run_output(
        foxes.models.outputs.FlowPlots2D(
            out_dir=tmp_path,
            plane="xy",
            reduction="mean",
            var=FV.WS,
            grid_pars=dict(n_img_points=(12, 10), z=90.0),
        ),
        farm_results=farm_results,
        algo=algo,
    )
    fpath = tmp_path / "flow_field_xy_smoke.png"
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)

    assert fpath.is_file()
    assert fpath.stat().st_size > 0

    fpath.unlink()
    assert not fpath.exists()
