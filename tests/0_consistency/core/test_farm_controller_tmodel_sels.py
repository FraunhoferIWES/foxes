import numpy as np

import foxes.constants as FC
from foxes.core import FarmController


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
        "data_vars": {FC.TMODEL_SELS: ((FC.STATE, FC.TURBINE, FC.TMODELS), np.ones((1, 1, 1), dtype=bool))},
        "extra_data": {},
    }

    # emulate post-load cleanup logic without invoking full model stack
    loaded_data["data_vars"][c._tmodel_sels_var(0)] = ((FC.STATE, FC.TURBINE), c._tmsels[0])
    loaded_data["data_vars"].pop(FC.TMODEL_SELS, None)

    assert FC.TMODEL_SELS not in loaded_data["data_vars"]
    assert c._tmodel_sels_var(0) in loaded_data["data_vars"]
