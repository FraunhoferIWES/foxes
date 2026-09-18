from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.algorithms.downwind.downwind import Downwind
from foxes.core import MData, States, TData
from foxes.input.states import BinnedStates, FieldData, PointCloudData


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

    def __init__(self):
        self.farm = SimpleNamespace(turbines=[SimpleNamespace(xy=np.array([0.0, 0.0]))])

    def new_point_data(self, points, n_states):
        return None


class _Downwind:
    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, force=False):
        pass


def _mock_source_results(n_points):
    return xr.Dataset(
        data_vars={
            FV.WS: (
                (FC.STATE, FC.POINT),
                np.tile([[2.0, 3.0, 7.0, 8.0]], (n_points, 1)).T,
            ),
            FV.WD: (
                (FC.STATE, FC.POINT),
                np.tile([[10.0, 190.0, 20.0, 200.0]], (n_points, 1)).T,
            ),
        }
    )


def _run_farm_with_states(states):
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(xy=[250.0, 250.0], H=90.0, turbine_models=["null_type"]),
        verbosity=0,
    )
    algo = Downwind(
        farm=farm,
        states=states,
        rotor_model="centre",
        wake_models=[],
        mbook=foxes.models.ModelBook(),
        verbosity=0,
    )
    with foxes.Engine.new("single", verbosity=0):
        return algo.calc_farm()


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
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)

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


def test_binned_states_grid_output_file_reruns_with_field_data(monkeypatch, tmp_path):
    source = _SourceStates()
    out_file = tmp_path / "binned_grid.nc"
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0], FV.WD: [0.0, 180.0, 360.0]},
        support_grid={
            FV.X: [0.0, 500.0],
            FV.Y: [0.0, 500.0],
            FV.H: [90.0],
        },
        output_file=out_file,
    )
    loaded_data = source.initialize(None)
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: _mock_source_results(4),
    )
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)
    monkeypatch.setattr(config, "_Config__utmn", None)
    monkeypatch.setattr(config, "_Config__utml", None)

    states.load_data(_Algorithm(), loaded_data)

    with xr.open_dataset(out_file, engine=config.nc_engine) as data:
        assert data.attrs["foxes_state_class"] == "FieldData"
        assert "utm_zone" not in data.attrs
        assert data[FV.WS].dims == ("Time", "height", "UTMY", "UTMX")

    field_states = FieldData(
        data_source=out_file,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.06, FV.RHO: 1.225},
        weight_ncvar=FV.WEIGHT,
        time_format=None,
        bounds_extra_space=None,
    )
    farm_results = _run_farm_with_states(field_states)
    assert farm_results.sizes[FC.STATE] == states.size()
    assert np.all(np.isfinite(farm_results[FV.AMB_REWS].to_numpy()))


def test_binned_states_output_file_writes_configured_utm_zone(monkeypatch, tmp_path):
    source = _SourceStates()
    out_file = tmp_path / "binned_utm.nc"
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0], FV.WD: [0.0, 180.0, 360.0]},
        support_points=np.array(
            [
                [0.0, 0.0, 90.0],
                [500.0, 0.0, 90.0],
                [0.0, 500.0, 90.0],
                [500.0, 500.0, 90.0],
            ]
        ),
        output_file=out_file,
    )
    loaded_data = source.initialize(None)
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: _mock_source_results(4),
    )
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)
    monkeypatch.setattr(config, "_Config__utmn", 33)
    monkeypatch.setattr(config, "_Config__utml", "U")

    states.load_data(_Algorithm(), loaded_data)

    with xr.open_dataset(out_file, engine=config.nc_engine) as data:
        assert data.attrs["utm_zone"] == "33U"


def test_binned_states_point_output_file_reruns_with_point_cloud_data(
    monkeypatch, tmp_path
):
    source = _SourceStates()
    out_file = tmp_path / "binned_points.nc"
    states = BinnedStates(
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0], FV.WD: [0.0, 180.0, 360.0]},
        support_points=np.array(
            [
                [0.0, 0.0, 90.0],
                [500.0, 0.0, 90.0],
                [0.0, 500.0, 90.0],
                [500.0, 500.0, 90.0],
            ]
        ),
        output_file=out_file,
    )
    loaded_data = source.initialize(None)
    monkeypatch.setattr(
        "foxes.input.states.binned_states.run_with_engine",
        lambda *args, **kwargs: _mock_source_results(4),
    )
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)
    monkeypatch.setattr(config, "_Config__utmn", None)
    monkeypatch.setattr(config, "_Config__utml", None)

    states.load_data(_Algorithm(), loaded_data)

    with xr.open_dataset(out_file, engine=config.nc_engine) as data:
        assert data.attrs["foxes_state_class"] == "PointCloudData"
        assert "utm_zone" not in data.attrs
        assert data[FV.WS].dims == ("Time", FC.POINT)
        assert data["height"].dims == (FC.POINT,)

    point_states = PointCloudData(
        data_source=out_file,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.06, FV.RHO: 1.225},
        weight_ncvar=FV.WEIGHT,
        h_ncvar="height",
        time_format=None,
    )
    farm_results = _run_farm_with_states(point_states)
    assert farm_results.sizes[FC.STATE] == states.size()
    assert np.all(np.isfinite(farm_results[FV.AMB_REWS].to_numpy()))


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
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)
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
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)
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
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)

    with pytest.raises(ValueError, match="state-dependent only"):
        states.load_data(_Algorithm(), loaded_data)
