import numpy as np
import pytest

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FarmController, FarmDataModelList, FData, MData
from foxes.models.turbine_models import Calculator


class _Model:
    def __init__(self, name):
        self.name = name


def test_get_pars_uses_full_slice_for_all_true_model_selection():
    c = FarmController()
    c.turbine_model_names = ["m0"]
    c._tmall = [True]

    pars = c._FarmController__get_pars(None, [_Model("m0")], "calc")

    s = pars[0]["st_sel"]
    assert isinstance(s, tuple)
    assert len(s) == 2
    assert all(x == slice(None) for x in s)


def test_get_pars_uses_model_specific_selection_variable():
    c = FarmController()
    c.turbine_model_names = ["m0", "m1"]
    c._tmall = [True, False]

    sel = np.array([[True, False], [False, True]])
    mdata = {c._tmodel_sels_var(1): sel}

    pars = c._FarmController__get_pars(None, [_Model("m1")], "calc", mdata=mdata)

    assert np.array_equal(pars[0]["st_sel"], sel)


def test_get_pars_uses_downwind_mask_for_model_specific_selection():
    c = FarmController()
    c.turbine_model_names = ["m0"]
    c._tmall = [False]

    sel = np.array(
        [
            [True, False, True],
            [False, False, True],
        ]
    )
    mdata = {c._tmodel_sels_var(0): sel}

    pars = c._FarmController__get_pars(
        None,
        [_Model("m0")],
        "calc",
        mdata=mdata,
        downwind_index=2,
    )

    s = pars[0]["st_sel"]
    assert isinstance(s, tuple)
    assert np.array_equal(s[0], sel[:, 2])
    assert s[1] == 2


def test_tmodel_sels_constant_is_no_longer_written_to_data_vars():
    c = FarmController()
    c.turbine_model_names = ["m0"]
    c._tmall = [False]
    c._tmsels = {0: np.array([[True]])}

    loaded_data = {
        "coords": {FC.TMODELS: np.array(["old"])},
        "data_vars": {
            FC.TMODEL_SELS: (
                (FC.STATE, FC.TURBINE, FC.TMODELS),
                np.ones((1, 1, 1), dtype=bool),
            )
        },
        "extra_data": {},
    }

    # emulate post-load cleanup logic without invoking full model stack
    loaded_data["data_vars"][c._tmodel_sels_var(0)] = (
        (FC.STATE, FC.TURBINE),
        c._tmsels[0],
    )
    loaded_data["data_vars"].pop(FC.TMODEL_SELS, None)

    assert FC.TMODEL_SELS not in loaded_data["data_vars"]
    assert c._tmodel_sels_var(0) in loaded_data["data_vars"]


@pytest.mark.parametrize(
    "names,expected",
    [
        (None, 8.0),
        (["second"], 6.0),
        (["first"], 4.0),
        (["second", "first"], 8.0),
        ([], 3.0),
    ],
)
def test_model_name_selection_preserves_order_and_skips_other_models(names, expected):
    first = Calculator([FV.P], [FV.P], lambda power, **_: (power + 1.0,))
    second = Calculator([FV.P], [FV.P], lambda power, **_: (power * 2.0,))
    first.name, second.name = "first", "second"
    models = FarmDataModelList([first, second])
    models.initialize(None, verbosity=0)
    fdata = FData(
        data={FV.P: np.array([[3.0]])},
        dims={FV.P: (FC.STATE, FC.TURBINE)},
    )

    results = models.calculate(None, MData(), fdata, model_names=names)

    np.testing.assert_array_equal(fdata[FV.P], [[expected]])
    assert set(results) == ({FV.P} if names is None or names else set())


def test_model_name_selection_rejects_unknown_names():
    models = FarmDataModelList([])
    models.initialize(None, verbosity=0)

    with pytest.raises(ValueError, match="Unknown model names.*unknown"):
        models.calculate(None, MData(), FData(), model_names=["unknown"])


def test_get_pars_aligns_original_farm_selection_without_mutating_model_data():
    controller = FarmController()
    controller.turbine_model_names = ["evaluation"]
    controller._tmall = [False]
    selection = np.array([[True, False, False], [True, False, False]])
    mdata = {controller._tmodel_sels_var(0): selection}
    order = np.array([[2, 0, 1], [1, 2, 0]])

    parameters = controller._FarmController__get_pars(
        None, [_Model("evaluation")], "calc", mdata=mdata, selection_order=order
    )

    np.testing.assert_array_equal(
        parameters[0]["st_sel"], [[False, True, False], [False, False, True]]
    )
    np.testing.assert_array_equal(
        mdata[controller._tmodel_sels_var(0)],
        [[True, False, False], [True, False, False]],
    )
