import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV


@pytest.fixture(params=[foxes.algorithms.Downwind, foxes.algorithms.Iterative])
def ambient_algorithm(request):
    farm = foxes.WindFarm()
    for position in ([0.0, 0.0], [630.0, 0.0]):
        farm.add_turbine(
            foxes.Turbine(xy=position, H=90.0, D=126.0, turbine_models=["NREL5MW"]),
            verbosity=0,
        )
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    return request.param(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k005"],
        verbosity=0,
    )


@pytest.mark.parametrize("ambient_keep", [None, False, True])
@pytest.mark.parametrize("calculation", ["farm", "points"])
def test_ambient_variable_retention(ambient_algorithm, calculation, ambient_keep):
    options = {} if ambient_keep is None else {"ambient_keep": ambient_keep}
    with foxes.Engine.new("single", verbosity=0):
        results = ambient_algorithm.calc_farm(ambient=True, **options)
        if calculation == "points":
            results = ambient_algorithm.calc_points(
                results, np.array([[800.0, 0.0, 90.0]]), ambient=True, **options
            )

    result_dimension = FC.TURBINE if calculation == "farm" else FC.POINT
    pairs = [
        (variable, ambient)
        for variable, ambient in FV.var2amb.items()
        if ambient in results
    ]
    assert pairs
    if calculation == "farm":
        assert FV.AMB_P in results
        assert results[FV.AMB_P].min() > 0.0
    for variable, ambient in pairs:
        assert results[ambient].dims == (FC.STATE, result_dimension)
        if ambient_keep:
            assert variable in results
            assert results[variable].dims == results[ambient].dims
            np.testing.assert_allclose(results[variable], results[ambient])
        else:
            assert variable not in results


@pytest.mark.parametrize("calculation", ["farm", "points"])
def test_ambient_keep_respects_outputs(ambient_algorithm, calculation):
    with foxes.Engine.new("single", verbosity=0):
        if calculation == "farm":
            results = ambient_algorithm.calc_farm(
                ambient=True, ambient_keep=True, outputs=[FV.P, FV.AMB_P]
            )
            assert FV.P in results and FV.AMB_P in results
            assert FV.WS not in results and FV.AMB_WS not in results
        else:
            farm_results = ambient_algorithm.calc_farm(ambient=True)
            results = ambient_algorithm.calc_points(
                farm_results,
                np.array([[800.0, 0.0, 90.0]]),
                ambient=True,
                ambient_keep=True,
                outputs=[FV.WS, FV.AMB_WS],
            )
            assert FV.WS in results and FV.AMB_WS in results
            assert FV.WD not in results and FV.AMB_WD not in results


@pytest.mark.parametrize("calculation", ["farm", "points"])
def test_ambient_keep_does_not_disable_wakes(ambient_algorithm, calculation):
    with foxes.Engine.new("single", verbosity=0):
        farm_results = ambient_algorithm.calc_farm()
        kept_results = ambient_algorithm.calc_farm(ambient_keep=True)
        if calculation == "farm":
            assert (farm_results[FV.P] < farm_results[FV.AMB_P]).any()
        else:
            points = np.array([[800.0, 0.0, 90.0]])
            kept_results = ambient_algorithm.calc_points(
                farm_results, points, ambient_keep=True
            )
            farm_results = ambient_algorithm.calc_points(farm_results, points)
            assert (farm_results[FV.WS] < farm_results[FV.AMB_WS]).any()
    xr.testing.assert_identical(farm_results, kept_results)


def test_ambient_keep_does_not_allow_wakes_without_farm_results(ambient_algorithm):
    with foxes.Engine.new("single", verbosity=0):
        with pytest.raises(ValueError, match="without farm results"):
            ambient_algorithm.calc_points(
                None, np.array([[800.0, 0.0, 90.0]]), ambient_keep=True
            )
