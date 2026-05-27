import numpy as np
from types import SimpleNamespace

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.kTI_model import kTI


def test_kti_calculate_selected_entries():
    model = kTI(kTI=0.3, kb=0.01, ti_var=FV.TI, k_var=FV.K)
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)

    st_sel = np.array([[True, False], [True, True]])
    fdata = FData(
        data={FV.K: np.zeros((2, 2), dtype=float)},
        dims={FV.K: (FC.STATE, FC.TURBINE)},
    )

    ti_sel = np.array([0.1, 0.2, 0.3], dtype=float)

    def fake_get_data(v, *_args, **_kwargs):
        if v == FV.KTI:
            return np.full(ti_sel.shape, 0.3)
        if v == FV.KB:
            return np.full(ti_sel.shape, 0.01)
        if v == FV.TI:
            return ti_sel
        raise KeyError(v)

    model.get_data = fake_get_data

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=st_sel)

    expected = np.zeros((2, 2), dtype=float)
    expected[st_sel] = 0.3 * ti_sel + 0.01
    assert np.allclose(out[FV.K], expected)


def main():
    test_kti_calculate_selected_entries()


if __name__ == "__main__":
    main()
