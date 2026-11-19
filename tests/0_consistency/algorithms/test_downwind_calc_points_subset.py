from _output_contract_helpers import run_output
import numpy as np
import pytest

import foxes
import foxes.variables as FV
import foxes.constants as FC


def test_downwind_calc_points_respects_states_sel_subset():
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[600.0, 0.0],
        n_turbines=1,
        turbine_models=["NREL5MW"],
        H=200.0,
        verbosity=0,
    )

    states = foxes.input.states.MultiHeightNCTimeseries(
        data_source="WRF-Timeseries-3000.nc",
        time_coord="Time",
        h_coord="height",
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"},
    )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="level10",
        wake_models=["Bastankhah2014_linear_k1_02"],
        partial_wakes=None,
        verbosity=0,
    )

    points = np.array([[[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]]], dtype=float)

    with foxes.Engine.new(engine_type=None):
        farm_results = algo.calc_farm(outputs="farm_results_all")
        point_results = algo.calc_points(
            farm_results,
            points=points,
            states_sel=["2009-01-06 13:50:00"],
        )
        with pytest.raises(TypeError, match="use state labels with 'states_sel'"):
            algo.calc_points(
                farm_results,
                points=points,
                states_isel=[0],
            )
        plot_data = run_output(
            foxes.models.outputs.FlowPlots2D(
                plane="xz",
                reduction="states",
                plot=False,
                data_format="numpy",
                var=FV.AMB_WS,
                states_sel=["2009-01-06 13:50:00"],
                grid_pars=dict(
                    n_img_points=(2, 2), xmin=0.0, xmax=100.0, zmin=50.0, zmax=500.0
                ),
            ),
            farm_results=farm_results,
            algo=algo,
        )

    assert point_results.sizes[FC.STATE] == 1
    assert point_results.sizes[FC.POINT] == 2
    assert plot_data[2].shape[0] == 1


if __name__ == "__main__":
    test_downwind_calc_points_respects_states_sel_subset()
