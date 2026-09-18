from types import SimpleNamespace

import numpy as np
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
import foxes.input.states.meso_micro_field as meso_micro_field_module
import foxes.input.states.ref_point_fields as ref_point_fields_module
from foxes.input.states import (
    FieldData,
    MesoMicroField,
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
