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
