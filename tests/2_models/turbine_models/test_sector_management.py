from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.sector_management import SectorManagement


def test_sector_management_sets_target_in_defined_ranges():
    df = pd.DataFrame(
        {
            "WD_min": [350.0, 90.0],
            "WD_max": [10.0, 210.0],
            FV.MAX_P: [1000.0, 2000.0],
        },
        index=pd.Index([0, 1], name=FC.TURBINE),
    )

    model = SectorManagement(
        data_source=df,
        range_vars=[FV.WD],
        target_vars=[FV.MAX_P],
        col_tinds=FC.TURBINE,
    )
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    fdata = FData(
        data={
            FV.WD: np.array([[355.0, 100.0], [5.0, 250.0]], dtype=float),
            FV.MAX_P: np.full((2, 2), np.nan, dtype=float),
        },
        dims={
            FV.WD: (FC.STATE, FC.TURBINE),
            FV.MAX_P: (FC.STATE, FC.TURBINE),
        },
    )
    st_sel = np.array([[True, True], [True, True]])

    out = model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(out[FV.MAX_P][0, 0], 1000.0)
    assert np.isclose(out[FV.MAX_P][1, 0], 1000.0)
    assert np.isclose(out[FV.MAX_P][0, 1], 2000.0)
    assert np.isnan(out[FV.MAX_P][1, 1])


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_sector_management_respects_turbine_identity_and_selection(ordered, selected):
    rules = pd.DataFrame(
        {"tname": ["WT01"], "WD_min": [340.0], "WD_max": [50.0], FV.MAX_P: [0.0]}
    )
    model = SectorManagement(
        rules, range_vars=[FV.WD], target_vars=[FV.MAX_P], col_tnames="tname"
    )
    algo = SimpleNamespace(
        farm=SimpleNamespace(turbine_names=["WT01", "WT02", "WT03"]),
        store_model_data=lambda *args, **kwargs: None,
    )
    model.initialize(algo=algo, verbosity=0)
    order = (
        np.array([[2, 0, 1], [1, 2, 0]]) if ordered else np.tile(np.arange(3), (2, 1))
    )
    data = {FV.WD: np.full((2, 3), 355.0), FV.MAX_P: np.full((2, 3), np.nan)}
    if ordered:
        data[FV.ORDER] = order
    fdata = FData(data=data, dims={name: (FC.STATE, FC.TURBINE) for name in data})
    st_sel = (order == 0) if selected else np.zeros((2, 3), dtype=bool)

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.where((order == 0) & st_sel, 0.0, np.nan)
    np.testing.assert_allclose(out[FV.MAX_P], expected)


@pytest.mark.parametrize(
    "algorithm", [foxes.algorithms.Downwind, foxes.algorithms.Iterative]
)
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_sector_management_shutdown_stays_on_named_turbine(algorithm, chunk_size):
    rules = pd.DataFrame(
        {
            "tname": ["WT01"],
            "WD_min": [80.0],
            "WD_max": [280.0],
            "REWS_min": [0.0],
            "REWS_max": [99.0],
            FV.MAX_P: [0.0],
        }
    )
    mbook = foxes.ModelBook()
    mbook.turbine_models["sector_rules"] = SectorManagement(
        rules, range_vars=[FV.WD, FV.REWS], target_vars=[FV.MAX_P], col_tnames="tname"
    )
    farm = foxes.WindFarm()
    for index in range(3):
        farm.add_turbine(
            foxes.Turbine(
                xy=[630.0 * index, 0.0],
                name=f"WT{index + 1:02d}",
                turbine_models=["NREL5MW", "sector_rules", "PMask"],
            ),
            verbosity=0,
        )
    states = foxes.input.states.StatesTable(
        pd.DataFrame({FV.WD: [90.0, 270.0]}),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.WS: 8.0, FV.TI: 0.08, FV.RHO: 1.225},
    )
    algo = algorithm(
        farm,
        states,
        mbook=mbook,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k005"],
        verbosity=0,
    )
    with foxes.Engine.new("single", chunk_size_states=chunk_size, verbosity=0):
        results = algo.calc_farm()

    assert results[FV.MAX_P].dims == (FC.STATE, FC.TURBINE)
    np.testing.assert_array_equal(results["tname"], ["WT01", "WT02", "WT03"])
    np.testing.assert_allclose(results[FV.MAX_P][:, 0], 0.0)
    assert np.isnan(results[FV.MAX_P][:, 1:]).all()
    np.testing.assert_allclose(results[FV.P][:, 0], 0.0)
    np.testing.assert_allclose(results[FV.CT][:, 0], 0.0)
    assert (results[FV.P][:, 1:] > 0.0).all()
    assert (results[FV.REWS][:, 1:] < results[FV.AMB_REWS][:, 1:]).any()


@pytest.mark.parametrize("invalid", ["unknown_turbine", "missing_range"])
def test_sector_management_rejects_invalid_rules(invalid):
    rules = pd.DataFrame(
        {"tname": ["WT01"], "WD_min": [0.0], "WD_max": [180.0], FV.MAX_P: [0.0]}
    )
    if invalid == "unknown_turbine":
        rules["tname"] = "unknown"
        error, message = ValueError, "unknown"
    else:
        rules = rules.drop(columns="WD_max")
        error, message = KeyError, "Missing column 'WD_max'"
    model = SectorManagement(
        rules, range_vars=[FV.WD], target_vars=[FV.MAX_P], col_tnames="tname"
    )
    algo = SimpleNamespace(
        farm=SimpleNamespace(turbine_names=["WT01"]),
        store_model_data=lambda *args, **kwargs: None,
    )
    with pytest.raises(error, match=message):
        model.initialize(algo=algo, verbosity=0)


def main():
    test_sector_management_sets_target_in_defined_ranges()


if __name__ == "__main__":
    main()
