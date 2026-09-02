import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV


@pytest.fixture(
    params=[
        (foxes.output.WindFarmsEval, FC.FARM),
        (foxes.output.ClusterEval, FC.CLUSTER),
    ]
)
def eval_case(request):
    return request.param


def _build_two_farm():
    farm = foxes.WindFarm()

    t0 = foxes.Turbine([0.0, 0.0], cluster_name="west")
    t0.wind_farm_name = "west"
    farm.add_turbine(t0, verbosity=0)

    t1 = foxes.Turbine([100.0, 0.0], cluster_name="east")
    t1.wind_farm_name = "east"
    farm.add_turbine(t1, verbosity=0)

    return farm


def _build_results(weight_data):
    return xr.Dataset(
        data_vars={
            FV.WEIGHT: ((FC.STATE, FC.TURBINE), weight_data),
        },
        coords={
            FC.STATE: np.array([0, 1], dtype=np.int32),
            FC.TURBINE: np.array([0, 1], dtype=np.int32),
        },
    )


def test_aggregate_uses_fallback_mapping_for_turbine_weights(eval_case):
    eval_cls, level = eval_case
    farm = _build_two_farm()
    farm_results = _build_results(
        np.array(
            [
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
    )

    out = eval_cls(farm, farm_results=farm_results)
    agg = out._aggregate(mapping=None)

    assert FV.WEIGHT in agg.data_vars
    assert agg[FV.WEIGHT].dims == (FC.STATE, level)
    for area_name in agg[level].values:
        w = agg[FV.WEIGHT].sel({level: area_name}).values
        np.testing.assert_allclose(w.sum(), 1.0)


def test_aggregate_rejects_zero_sum_turbine_weights(eval_case):
    eval_cls, _ = eval_case
    farm = _build_two_farm()
    farm_results = _build_results(np.zeros((2, 2), dtype=np.float64))

    out = eval_cls(farm, farm_results=farm_results)

    with pytest.raises(ValueError, match="Cannot normalize"):
        out._aggregate(mapping=None)


def test_farm_eval_allows_missing_farm_results():
    farm = _build_two_farm()
    out = foxes.output.WindFarmsEval(farm=farm, farm_results=None)

    assert out.farm_results is None
    assert out.get_mapping() == {"west": [0], "east": [1]}

    with pytest.raises(AssertionError, match="farm_results are required"):
        out._aggregate(mapping=None)


def test_map_turbines_to_areas_preserves_clusters_unless_forced():
    farm = _build_two_farm()
    farm.add_turbine(foxes.Turbine([50.0, 0.0]), verbosity=0)
    areas = {"all": foxes.utils.geom2d.Circle([50.0, 0.0], 100.0)}

    mapping = farm.map_turbines_to_areas(areas, farm_single_cluster=False)

    assert mapping == {"all": [0, 1, 2]}
    assert farm.cluster_list == ["west", "east", "all"]

    mapping = farm.map_turbines_to_areas(areas, force=True)

    assert mapping == {"all": [0, 1, 2]}
    assert farm.cluster_list == ["all", "all", "all"]


def test_map_turbines_to_areas_uses_wind_farm_area_majority():
    farm = foxes.WindFarm()
    turbines = [
        ([0.0, 0.0], "previous"),
        ([10.0, 0.0], None),
        ([100.0, 0.0], None),
    ]
    for xy, cluster_name in turbines:
        farm.add_turbine(
            foxes.Turbine(xy, wind_farm_name="farm", cluster_name=cluster_name),
            verbosity=0,
        )
    farm.add_turbine(foxes.Turbine([50.0, 100.0], wind_farm_name="farm"), verbosity=0)
    areas = {
        "west": foxes.utils.geom2d.Circle([5.0, 0.0], 25.0),
        "east": foxes.utils.geom2d.Circle([100.0, 0.0], 25.0),
    }

    mapping = farm.map_turbines_to_areas(areas)

    assert mapping == {"west": [0, 1], "east": [2]}
    assert farm.cluster_list == ["west", "west", "west", "west"]


def test_map_turbines_to_areas_keeps_unmapped_farm_clusters():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([100.0, 0.0], cluster_name="previous"), verbosity=0)
    areas = {"west": foxes.utils.geom2d.Circle([0.0, 0.0], 25.0)}

    mapping = farm.map_turbines_to_areas(areas)

    assert mapping == {"west": []}
    assert farm.cluster_list == ["previous"]
