import numpy as np
import pytest
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import MData, States, TData
from foxes.input.states import BinnedStates


class _SourceStates(States):
    def __init__(self):
        super().__init__()

    def size(self):
        return 4

    def output_point_vars(self, algo):
        return [FV.WS, FV.WD]

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        loaded_data["coords"][FC.STATE] = np.arange(4)
        loaded_data["data_vars"][FV.WEIGHT] = (
            (FC.STATE,),
            np.array([1.0, 2.0, 1.0, 2.0]),
        )

    def calculate(self, algo, mdata, fdata, tdata):
        return {}


class _Algorithm:
    n_states = None

    def new_point_data(self, points, n_states):
        return None


def test_binned_states_reduces_into_loaded_data(monkeypatch):
    source = _SourceStates()
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0], FV.WD: [0.0, 180.0, 360.0]},
        support_points=np.array([[0.0, 0.0, 100.0], [1.0, 0.0, 100.0]]),
    )
    loaded_data = source.initialize(None)

    source_results = xr.Dataset(
        data_vars={
            FV.WS: (
                (FC.STATE, FC.POINT),
                [[2.0, 7.0], [3.0, 8.0], [6.0, 9.0], [7.0, 4.0]],
            ),
            FV.WD: (
                (FC.STATE, FC.POINT),
                [[359.0, 1.0], [10.0, 20.0], [190.0, 200.0], [350.0, 170.0]],
            ),
        }
    )
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: source_results,
    )

    states.load_data(_Algorithm(), loaded_data)

    assert states.size() == 4
    assert loaded_data["data_vars"][states.var(FV.WS)][0] == (FC.STATE, FC.POINT)
    assert loaded_data["data_vars"][states.var(FV.WEIGHT)][0] == (
        FC.STATE,
        FC.POINT,
    )
    assert states._source_state_key in loaded_data["coords"]
    np.testing.assert_allclose(
        loaded_data["data_vars"][states.var(FV.WEIGHT)][1],
        [[2.0, 2.0], [1.0, 0.0], [0.0, 3.0], [3.0, 1.0]],
    )


def test_binned_states_ignores_height_for_single_height_support(monkeypatch):
    source = _SourceStates()
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 10.0]},
        support_points=np.array(
            [[0.0, 0.0, 100.0], [1.0, 0.0, 100.0], [0.0, 1.0, 100.0]]
        ),
    )
    loaded_data = source.initialize(None)
    source_results = xr.Dataset(
        data_vars={
            FV.WS: ((FC.STATE, FC.POINT), np.tile([[0.0, 1.0, 2.0]], (4, 1))),
            FV.WD: ((FC.STATE, FC.POINT), np.ones((4, 3))),
        }
    )
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: source_results,
    )
    states.load_data(_Algorithm(), loaded_data)

    mdata = MData(
        data={
            FC.STATE: np.array([0]),
            states.var(FV.WS): loaded_data["data_vars"][states.var(FV.WS)][1],
            states.var(FV.WEIGHT): loaded_data["data_vars"][states.var(FV.WEIGHT)][1],
        },
        dims={
            FC.STATE: (FC.STATE,),
            states.var(FV.WS): (FC.STATE, FC.POINT),
            states.var(FV.WEIGHT): (FC.STATE, FC.POINT),
        },
        extra_data=loaded_data["extra_data"],
        states_i0=0,
    )
    tdata = TData.from_tpoints(
        np.array([[[[0.5, 0.0, 0.0]], [[0.5, 0.0, 999.0]]]]),
        np.array([1.0]),
        mdata=mdata,
    )
    result = states.calculate(None, mdata, None, tdata)

    np.testing.assert_allclose(result[FV.WS][0, :, 0], [0.5, 0.5])


def test_binned_states_wraps_wind_direction_at_north(monkeypatch):
    source = _SourceStates()
    states = BinnedStates(
        source,
        bin_vars={FV.WD: [350.0, 370.0]},
        support_points=np.array([[0.0, 0.0, 100.0]]),
    )
    loaded_data = source.initialize(None)
    source_results = xr.Dataset(
        data_vars={
            FV.WS: ((FC.STATE, FC.POINT), np.ones((4, 1))),
            FV.WD: ((FC.STATE, FC.POINT), [[359.0], [0.0], [1.0], [180.0]]),
        }
    )
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: source_results,
    )
    states.load_data(_Algorithm(), loaded_data)

    weights = loaded_data["data_vars"][states.var(FV.WEIGHT)][1]
    directions = loaded_data["data_vars"][states.var(FV.WD)][1]
    np.testing.assert_allclose(weights[:, 0], [4.0])
    np.testing.assert_allclose(directions[:, 0], [0.0], atol=1e-12)


def test_binned_states_rejects_point_dependent_weights(monkeypatch):
    source = _SourceStates()
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 10.0]},
        support_points=np.array([[0.0, 0.0, 100.0]]),
    )
    loaded_data = source.initialize(None)
    loaded_data["data_vars"][FV.WEIGHT] = (
        (FC.STATE, FC.POINT),
        np.ones((4, 1)),
    )
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: xr.Dataset(
            data_vars={
                FV.WS: ((FC.STATE, FC.POINT), np.ones((4, 1))),
            }
        ),
    )

    with pytest.raises(ValueError, match="state-dependent only"):
        states.load_data(_Algorithm(), loaded_data)
