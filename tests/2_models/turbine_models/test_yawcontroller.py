from types import SimpleNamespace

import numpy as np
import pytest

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_models.yawcontroller import YawController


class DummySeries:
    def __init__(self, data):
        self._data = np.asarray(data)

    def to_numpy(self):
        return self._data


def _build_algo(counter=0):
    times = np.array(
        ["2020-01-01T00:00:00", "2020-01-01T00:01:00"], dtype="datetime64[s]"
    )
    states = SimpleNamespace(index=lambda: times)
    farm_results = {
        FV.YAW: DummySeries(np.zeros((2, 2), dtype=float)),
        FV.AMB_WD: DummySeries(np.zeros((2, 2), dtype=float)),
        FV.AMB_REWS: DummySeries(np.ones((2, 2), dtype=float)),
    }
    return SimpleNamespace(
        n_turbines=2,
        states=states,
        counter=counter,
        farm_results_downwind=farm_results,
        store_model_data=lambda *args, **kwargs: None,
    )


def test_yawcontroller_first_step_sets_zero_yawm():
    model = YawController(max_yaw_rate=0.2, max_yawm=7.5, avg_time=60)
    algo = _build_algo(counter=0)
    model.initialize(algo=algo, verbosity=0)

    fdata = FData(
        data={
            FV.AMB_WD: np.array([[15.0, 30.0]], dtype=float),
            FV.AMB_REWS: np.array([[8.0, 9.0]], dtype=float),
            FV.YAW: np.array([[20.0, 35.0]], dtype=float),
            FV.YAWM: np.array([[1.0, -2.0]], dtype=float),
        },
        dims={
            FV.AMB_WD: (FC.STATE, FC.TURBINE),
            FV.AMB_REWS: (FC.STATE, FC.TURBINE),
            FV.YAW: (FC.STATE, FC.TURBINE),
            FV.YAWM: (FC.STATE, FC.TURBINE),
        },
    )

    out = model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=np.s_[:])
    assert np.allclose(out[FV.YAWM], np.zeros((1, 2), dtype=float))


def test_yawcontroller_requires_single_state():
    model = YawController()
    algo = _build_algo(counter=0)
    model.initialize(algo=algo, verbosity=0)

    fdata = FData(
        data={
            FV.AMB_WD: np.zeros((2, 2), dtype=float),
            FV.AMB_REWS: np.ones((2, 2), dtype=float),
            FV.YAW: np.zeros((2, 2), dtype=float),
            FV.YAWM: np.zeros((2, 2), dtype=float),
        },
        dims={
            FV.AMB_WD: (FC.STATE, FC.TURBINE),
            FV.AMB_REWS: (FC.STATE, FC.TURBINE),
            FV.YAW: (FC.STATE, FC.TURBINE),
            FV.YAWM: (FC.STATE, FC.TURBINE),
        },
    )

    with pytest.raises(AssertionError, match="Sequential"):
        model.calculate(algo=algo, mdata={}, fdata=fdata, st_sel=np.s_[:])


def main():
    test_yawcontroller_first_step_sets_zero_yawm()
    test_yawcontroller_requires_single_state()


if __name__ == "__main__":
    main()
