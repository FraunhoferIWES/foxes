from types import SimpleNamespace

import numpy as np
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
import foxes.input.states.field_data as field_data_module
import foxes.input.states.meso_micro_field as meso_micro_field_module
import foxes.input.states.ref_point_fields as ref_point_fields_module
from foxes.core import MData, TData
from foxes.input.states import (
    FieldData,
    ICONStates,
    MesoMicroField,
    NEWAStates,
    PointCloudData,
    SectorSimRefPointField,
    SingleStateStates,
)


class _PlotTriggered(Exception):
    pass


def _make_fake_downwind(loaded_data):
    class _FakeDownwind:
        def __init__(self, *args, **kwargs):
            self.loaded_data = loaded_data

        def initialize(self, *args, **kwargs):
            pass

        def init_states(self, *args, **kwargs):
            pass

    return _FakeDownwind


def _make_ref_point_field(support_point_plot=None):
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[400.0, 0.0],
        n_turbines=2,
        turbine_models=["NREL5MW"],
        H=90.0,
        verbosity=0,
    )
    field_states = FieldData(
        "unused.nc",
        output_vars=[FV.WS, FV.WD],
        bounds_extra_space=None,
    )
    ref_point_states = SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    states = SectorSimRefPointField(
        field_states=field_states,
        ref_point_states=ref_point_states,
        ref_point=[200.0, 50.0, 100.0],
        support_point_plot=support_point_plot,
    )
    loaded_data = {
        "coords": {
            field_states.var(FV.X): np.array([-100.0, 200.0, 500.0]),
            field_states.var(FV.Y): np.array([-200.0, 50.0, 300.0]),
            field_states.var(FV.H): np.array([80.0, 100.0]),
            FC.STATE: np.array([0]),
        },
        "data_vars": {},
        "extra_data": {},
    }
    algo = SimpleNamespace(farm=farm, loaded_data=loaded_data)
    return states, algo, loaded_data


def test_sector_sim_ref_point_field_writes_support_point_plot(tmp_path):
    fpath = tmp_path / "support_points.png"
    states, algo, loaded_data = _make_ref_point_field(support_point_plot=str(fpath))

    states.write_support_point_plot(algo=algo, loaded_data=loaded_data)

    assert fpath.is_file()
    assert fpath.stat().st_size > 0


def test_sector_sim_ref_point_field_support_plot_requires_file_name():
    states, algo, loaded_data = _make_ref_point_field()

    with pytest.raises(ValueError, match="Missing file_name"):
        states.write_support_point_plot(algo=algo, loaded_data=loaded_data)


def test_sector_sim_ref_point_field_load_data_triggers_support_point_plot(monkeypatch):
    states, algo, _ = _make_ref_point_field(support_point_plot="support_points.png")
    field_loaded_data = {
        "coords": {
            FC.STATE: np.array([0]),
            states.field_states.var(FV.X): np.array([0.0, 100.0]),
            states.field_states.var(FV.Y): np.array([0.0, 100.0]),
            states.field_states.var(FV.H): np.array([100.0]),
        },
        "data_vars": {},
        "extra_data": {},
    }

    def _write_support_point_plot(self, algo, loaded_data, file_name=None, **kwargs):
        assert file_name == "support_points.png"
        assert states.field_states.var(FV.X) in loaded_data["coords"]
        raise _PlotTriggered

    monkeypatch.setattr(
        ref_point_fields_module, "Downwind", _make_fake_downwind(field_loaded_data)
    )
    monkeypatch.setattr(
        SectorSimRefPointField, "write_support_point_plot", _write_support_point_plot
    )

    with pytest.raises(_PlotTriggered):
        states.load_data(algo, {"coords": {}, "data_vars": {}, "extra_data": {}})


def _make_meso_micro_field(support_point_plot=None):
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=[0.0, 0.0],
        xy_step=[400.0, 0.0],
        n_turbines=2,
        turbine_models=["NREL5MW"],
        H=90.0,
        verbosity=0,
    )
    micro_states = FieldData(
        "unused_micro.nc",
        output_vars=[FV.WS, FV.WD],
        bounds_extra_space=None,
    )
    meso_states = FieldData(
        "unused_meso.nc",
        output_vars=[FV.WS, FV.WD],
        bounds_extra_space=None,
    )
    states = MesoMicroField(
        micro_states=micro_states,
        meso_states=meso_states,
        ref_points=np.array([[200.0, 50.0, 100.0], [600.0, 200.0, 100.0]]),
        support_point_plot=support_point_plot,
    )
    loaded_data = {
        "coords": {
            micro_states.var(FV.X): np.array([-100.0, 200.0, 500.0]),
            micro_states.var(FV.Y): np.array([-200.0, 50.0, 300.0]),
            micro_states.var(FV.H): np.array([80.0, 100.0]),
            FC.STATE: np.array([0]),
        },
        "data_vars": {},
        "extra_data": {},
    }
    algo = SimpleNamespace(farm=farm, loaded_data=loaded_data)
    return states, algo, loaded_data


def test_meso_micro_field_writes_support_point_plot(tmp_path):
    fpath = tmp_path / "meso_micro_support_points.png"
    states, algo, loaded_data = _make_meso_micro_field(support_point_plot=str(fpath))

    states.write_support_point_plot(algo=algo, loaded_data=loaded_data)

    assert fpath.is_file()
    assert fpath.stat().st_size > 0


def test_meso_micro_field_support_plot_requires_file_name():
    states, algo, loaded_data = _make_meso_micro_field()

    with pytest.raises(ValueError, match="Missing file_name"):
        states.write_support_point_plot(algo=algo, loaded_data=loaded_data)


def test_meso_micro_field_load_data_triggers_support_point_plot(monkeypatch):
    states, algo, _ = _make_meso_micro_field(support_point_plot="support_points.png")
    micro_loaded_data = {
        "coords": {
            FC.STATE: np.array([0]),
            states.micro_states.var(FV.X): np.array([0.0, 100.0]),
            states.micro_states.var(FV.Y): np.array([0.0, 100.0]),
            states.micro_states.var(FV.H): np.array([100.0]),
        },
        "data_vars": {},
        "extra_data": {},
    }

    def _write_support_point_plot(self, algo, loaded_data, file_name=None, **kwargs):
        assert file_name == "support_points.png"
        assert states.micro_states.var(FV.X) in loaded_data["coords"]
        assert states.REF_POINTS in loaded_data["data_vars"]
        raise _PlotTriggered

    monkeypatch.setattr(
        meso_micro_field_module, "Downwind", _make_fake_downwind(micro_loaded_data)
    )
    monkeypatch.setattr(
        MesoMicroField, "write_support_point_plot", _write_support_point_plot
    )

    with pytest.raises(_PlotTriggered):
        states.load_data(algo, {"coords": {}, "data_vars": {}, "extra_data": {}})


@pytest.mark.parametrize(
    ("ref_height", "expected"),
    [
        (None, [[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]]),
        (80.0, [[0.0, 0.0, 80.0]]),
    ],
)
def test_meso_micro_field_selects_point_cloud_reference_height(ref_height, expected):
    states, _, _ = _make_meso_micro_field()
    meso_states = PointCloudData(
        "unused.nc",
        output_vars=[FV.WS, FV.WD],
        states_coord=FC.STATE,
        point_coord=FC.POINT,
        x_ncvar=FV.X,
        y_ncvar=FV.Y,
        h_ncvar=FV.H,
    )
    support = np.array([[0.0, 0.0, 80.0], [0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    states.meso_states = meso_states
    states.ref_height = ref_height
    loaded_data = {
        "coords": {
            meso_states.var(FC.POINT): ((FC.POINT, FC.XYH), support),
        },
        "data_vars": {},
        "extra_data": {},
    }

    ref_points = states._get_default_ref_points(loaded_data)

    np.testing.assert_allclose(ref_points, expected)
    assert states.ref_height == (100.0 if ref_height is None else ref_height)


def test_meso_micro_field_gets_field_data_reference_points():
    states, _, _ = _make_meso_micro_field()
    meso_states = FieldData(
        "unused.nc",
        output_vars=[FV.WS, FV.WD],
        states_coord=FC.STATE,
        x_coord=FV.X,
        y_coord=FV.Y,
        h_coord=FV.H,
        time_format=None,
    )
    states.meso_states = meso_states
    states.ref_height = 80.0
    loaded_data = {
        "coords": {
            meso_states.var(FV.X): np.array([0.0, 100.0]),
            meso_states.var(FV.Y): np.array([0.0, 50.0]),
            meso_states.var(FV.H): np.array([80.0, 100.0]),
        },
        "data_vars": {},
        "extra_data": {},
    }

    ref_points = states._get_default_ref_points(loaded_data)

    np.testing.assert_allclose(
        ref_points,
        [[0.0, 0.0, 80.0], [0.0, 50.0, 80.0], [100.0, 0.0, 80.0], [100.0, 50.0, 80.0]],
    )


def test_meso_micro_field_gets_newa_reference_points():
    states, _, _ = _make_meso_micro_field()
    meso_states = NEWAStates("unused.nc", output_vars=[FV.WS, FV.WD])
    meso_states.XY = meso_states.var(f"{FV.X}{FV.Y}")
    meso_states.H = meso_states.var(FV.H)
    states.meso_states = meso_states
    states.ref_height = 90.0
    xy = np.array(
        [
            [[0.0, 0.0], [0.0, 50.0]],
            [[100.0, 0.0], [100.0, 50.0]],
        ]
    )
    loaded_data = {
        "coords": {meso_states.H: np.array([80.0, 100.0])},
        "data_vars": {meso_states.XY: ((FV.X, FV.Y, FC.XY), xy)},
        "extra_data": {},
    }

    ref_points = states._get_default_ref_points(loaded_data)

    np.testing.assert_allclose(
        ref_points,
        [[0.0, 0.0, 90.0], [0.0, 50.0, 90.0], [100.0, 0.0, 90.0], [100.0, 50.0, 90.0]],
    )


def test_meso_micro_field_gets_projected_icon_reference_points(monkeypatch):
    states, _, _ = _make_meso_micro_field()
    meso_states = ICONStates(
        "unused.nc",
        output_vars=[FV.WS, FV.WD],
        height_coord_tke=None,
    )
    states.meso_states = meso_states
    states.ref_height = 100.0
    loaded_data = {
        "coords": {
            meso_states.var(FV.X): np.array([8.0, 8.1]),
            meso_states.var(FV.Y): np.array([53.0]),
            meso_states.var(FV.H): np.array([80.0, 100.0]),
        },
        "data_vars": {},
        "extra_data": {},
    }
    monkeypatch.setattr(
        field_data_module,
        "from_lonlat",
        lambda points: points + np.array([1000.0, 2000.0]),
    )

    ref_points = states._get_default_ref_points(loaded_data)

    np.testing.assert_allclose(
        ref_points,
        [[1008.0, 2053.0, 100.0], [1008.1, 2053.0, 100.0]],
    )


def test_meso_micro_field_interpolates_ref_weights_independently_of_meso_model():
    states, _, _ = _make_meso_micro_field()
    ref_points = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]], dtype=float)
    targets = np.array(
        [
            [[[0.0, 25.0, 90.0], [25.0, -10.0, 90.0]]],
            [[[75.0, 10.0, 90.0], [100.0, -25.0, 90.0]]],
        ],
        dtype=float,
    )

    class _TData:
        n_targets = 1
        n_tpoints = 2

        def __getitem__(self, key):
            if key == FC.TARGETS:
                return targets
            raise KeyError(key)

    refw = states._interpolate_ref_weights(
        tdata=_TData(),
        ref_points=ref_points,
        n_states=2,
        n_tpts=2,
    )

    np.testing.assert_allclose(
        refw,
        [
            [[1.0, 0.0], [0.75, 0.25]],
            [[0.25, 0.75], [0.0, 1.0]],
        ],
    )


def test_meso_micro_field_uses_meso_weights_at_final_targets():
    states, _, _ = _make_meso_micro_field()
    ref_points = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    targets = np.array(
        [
            [[[25.0, 0.0, 90.0], [75.0, 0.0, 90.0]]],
            [[[30.0, 0.0, 90.0], [80.0, 0.0, 90.0]]],
        ]
    )
    mdata = MData(
        data={FC.STATE: np.array([0, 1])},
        dims={FC.STATE: (FC.STATE,)},
    )
    tdata = TData.from_tpoints(
        tpoints=targets,
        tweights=np.array([0.5, 0.5]),
        mdata=mdata,
    )

    class _MesoStates:
        def calculate(self, algo, mdata, fdata, tdata):
            x = tdata[FC.TARGETS][..., 0]
            tdata[FV.WEIGHT] = x / 100.0
            tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
            return {
                FV.WS: np.full_like(x, 8.0),
                FV.WD: np.full_like(x, 270.0),
            }

    states.meso_states = _MesoStates()
    results = states._calculate_meso_data(
        algo=None,
        mdata=mdata,
        fdata=None,
        tdata=tdata,
        ref_points=ref_points,
    )

    np.testing.assert_allclose(results[FV.WS], 8.0)
    np.testing.assert_allclose(results[FV.WD], 270.0)
    np.testing.assert_allclose(
        tdata[FV.WEIGHT],
        [[[0.25, 0.75]], [[0.3, 0.8]]],
    )


def test_meso_micro_field_preserves_state_only_meso_weights():
    states, _, _ = _make_meso_micro_field()
    ref_points = np.array([[0.0, 0.0, 100.0], [100.0, 0.0, 100.0]])
    targets = np.array(
        [
            [[[25.0, 0.0, 90.0], [75.0, 0.0, 90.0]]],
            [[[30.0, 0.0, 90.0], [80.0, 0.0, 90.0]]],
        ]
    )
    mdata = MData(
        data={FC.STATE: np.array([0, 1])},
        dims={FC.STATE: (FC.STATE,)},
    )
    tdata = TData.from_tpoints(
        tpoints=targets,
        tweights=np.array([0.5, 0.5]),
        mdata=mdata,
    )

    class _MesoStates:
        def calculate(self, algo, mdata, fdata, tdata):
            shape = tdata[FC.TARGETS].shape[:-1]
            tdata[FV.WEIGHT] = np.array([0.25, 0.75])[:, None, None]
            tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
            return {
                FV.WS: np.full(shape, 8.0),
                FV.WD: np.full(shape, 270.0),
            }

    states.meso_states = _MesoStates()
    states._calculate_meso_data(
        algo=None,
        mdata=mdata,
        fdata=None,
        tdata=tdata,
        ref_points=ref_points,
    )

    assert tdata[FV.WEIGHT].shape == (2, 1, 1)
    np.testing.assert_allclose(tdata[FV.WEIGHT][:, 0, 0], [0.25, 0.75])
