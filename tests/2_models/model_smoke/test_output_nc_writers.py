import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
