from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.core import FData, MData, States, TData
from foxes.input.states import MesoMicroField
from foxes.input.states.binned import (
    BinnedFieldData,
    BinnedPointCloudData,
    read_binned_data,
)
from foxes.input.states.field_data import FieldData
from foxes.input.states.point_cloud_data import PointCloudData


_SUPPORT_POINTS = np.array(
    [
        [0.0, 0.0, 90.0],
        [500.0, 0.0, 90.0],
        [0.0, 500.0, 90.0],
        [500.0, 500.0, 90.0],
    ]
)
_SUPPORT_GRID = {
    FV.X: [0.0, 500.0],
    FV.Y: [0.0, 500.0],
    FV.H: [90.0],
}


class _SourceStates(States):
    def size(self):
        return 4

    def output_point_vars(self, algo):
        return [FV.WS, FV.WD, FV.RHO]

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
        turbine = SimpleNamespace(xy=np.array([0.0, 0.0]))
        self.farm = SimpleNamespace(turbines=[turbine])


class _Downwind:
    source_results = None
    outputs = None
    ambient = None
    farm_ambient = None

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, force=False):
        pass

    def calc_farm(self, ambient=False):
        self.__class__.farm_ambient = ambient
        return xr.Dataset()

    def calc_points(self, farm_results, points, outputs, ambient=False):
        self.__class__.outputs = outputs
        self.__class__.ambient = ambient
        return self.__class__.source_results


def _source_results(n_points, *, variables=(FV.WS, FV.WD, FV.RHO)):
    values = {
        FV.WS: [2.0, 3.0, 7.0, 8.0],
        FV.WD: [10.0, 190.0, 20.0, 200.0],
        FV.RHO: [1.0, 1.2, 1.4, 1.6],
    }
    return xr.Dataset(
        data_vars={
            variable: (
                (FC.STATE, FC.POINT),
                np.broadcast_to(
                    np.asarray(values[variable])[:, None],
                    (4, n_points),
                ).copy(),
            )
            for variable in variables
        }
    )


def _patch_source_evaluation(monkeypatch, source_results):
    _Downwind.source_results = source_results
    _Downwind.outputs = None
    _Downwind.ambient = None
    _Downwind.farm_ambient = None
    monkeypatch.setattr(
        "foxes.input.states.binned._reduction.run_with_engine",
        lambda function, *args, **kwargs: function(),
    )
    monkeypatch.setattr("foxes.algorithms.Downwind", _Downwind)


def _source_binned(states_class, source, **kwargs):
    support = (
        {"support_grid": _SUPPORT_GRID}
        if states_class is BinnedFieldData
        else {"support_points": _SUPPORT_POINTS}
    )
    return states_class(source, **support, **kwargs)


def _meso_binned_artifact(states_class):
    writer = _source_binned(
        states_class,
        _SourceStates(),
        bin_vars={
            FV.WS: [0.0, 5.0, 10.0],
            FV.WD: [0.0, 180.0, 360.0],
        },
        mean_vars=[],
    )
    writer._binned.calculation_vars(None)
    support, axes = writer._binned._materialize_support()
    weights = np.zeros((4, len(support)))
    at_right = support[:, 0] == 500.0
    weights[1] = np.where(at_right, 0.8, 0.2)
    weights[2] = np.where(at_right, 0.2, 0.8)
    ws_mean = np.broadcast_to(
        np.array([2.0, 4.0, 8.0, 9.0])[:, None], weights.shape
    ).copy()
    wd_mean = np.broadcast_to(
        np.array([20.0, 200.0, 40.0, 220.0])[:, None], weights.shape
    ).copy()
    stats = {
        FV.WS: {name: ws_mean.copy() for name in ("min", "mean", "max")},
        FV.WD: {name: wd_mean.copy() for name in ("min", "mean", "max")},
    }
    return writer._binned._create_output_dataset(support, axes, stats, weights)


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_is_meso_micro_compatible(states_class):
    meso_states = states_class(_meso_binned_artifact(states_class))
    loaded_data = meso_states.initialize(None)
    states = MesoMicroField(
        micro_states=meso_states,
        meso_states=meso_states,
    )
    ref_points = states._get_default_ref_points(loaded_data)

    dataset = xr.Dataset(
        coords=loaded_data["coords"],
        data_vars=loaded_data["data_vars"],
    )
    mdata = MData.from_dataset(
        dataset,
        extra_data=loaded_data["extra_data"],
        states_i0=0,
    )
    fdata = FData.from_sizes(n_states=2, n_turbines=1)
    targets = np.array(
        [
            [[[125.0, 250.0, 90.0], [375.0, 250.0, 90.0]]],
            [[[125.0, 250.0, 90.0], [375.0, 250.0, 90.0]]],
        ]
    )
    tdata = TData.from_tpoints(
        tpoints=targets,
        tweights=np.array([0.5, 0.5]),
        mdata=mdata,
    )

    ref_results = states._calculate_meso_data(
        algo=None,
        mdata=mdata,
        fdata=fdata,
        tdata=tdata,
        ref_points=ref_points,
    )

    np.testing.assert_array_equal(mdata[FC.STATE], [1, 2])
    assert ref_results[FV.WS].shape == (2, 4)
    assert ref_results[FV.WD].shape == (2, 4)
    np.testing.assert_allclose(
        tdata[FV.WEIGHT],
        [[[0.35, 0.65]], [[0.65, 0.35]]],
    )


@pytest.mark.parametrize(
    ("states_class", "base_class", "support_dims"),
    [
        (BinnedFieldData, FieldData, (FV.X, FV.Y, FV.H)),
        (BinnedPointCloudData, PointCloudData, (FC.POINT,)),
    ],
    ids=["field", "point-cloud"],
)
def test_binned_data_reduces_and_round_trips_artifact(
    monkeypatch,
    tmp_path,
    states_class,
    base_class,
    support_dims,
):
    assert states_class.__bases__ == (base_class,)

    source = _SourceStates()
    output_file = tmp_path / f"{states_class.__name__}.nc"
    states = _source_binned(
        states_class,
        source,
        bin_vars={
            FV.WS: [0.0, 5.0, 10.0],
            FV.WD: [0.0, 180.0, 360.0],
        },
        output_file=output_file,
    )
    loaded_data = source.initialize(None)
    _patch_source_evaluation(monkeypatch, _source_results(4))

    states.load_data(_Algorithm(), loaded_data)

    assert _Downwind.outputs == [FV.AMB_WS, FV.AMB_WD, FV.AMB_RHO]
    assert _Downwind.ambient is True
    assert _Downwind.farm_ambient is True
    assert states.mean_vars == [FV.RHO]
    assert states.output_point_vars(None) == [FV.WS, FV.WD, FV.RHO]
    assert states.index() == [0, 1, 2, 3]

    with xr.open_dataset(output_file, engine=config.nc_engine) as data:
        assert data.attrs["foxes_state_class"] == states_class.__name__
        assert FV.WS not in data
        assert FV.WD not in data
        assert "bin_centres" not in data
        assert "binned_state_var" not in data
        assert data[FV.RHO].dims == (FC.STATE,)
        assert data[FV.WEIGHT].dims == (FC.STATE, *support_dims)
        np.testing.assert_array_equal(data[FC.STATE], [0, 1, 2, 3])
        np.testing.assert_allclose(
            data[FV.WEIGHT].to_numpy().reshape(4, -1),
            np.broadcast_to([1.0, 2.0, 1.0, 2.0], (4, 4)).T,
        )
        np.testing.assert_allclose(data[FV.RHO], [1.0, 1.2, 1.4, 1.6])
        for variable in (FV.WS, FV.WD, FV.RHO):
            assert f"{variable}_mean" not in data
            assert f"{variable}_std" not in data
        if states_class is BinnedFieldData:
            for coordinate, values in _SUPPORT_GRID.items():
                np.testing.assert_allclose(data[coordinate], values)
        else:
            np.testing.assert_allclose(data["support"], _SUPPORT_POINTS)

    reloaded = read_binned_data(output_file)
    assert type(reloaded) is states_class
    reloaded_data = reloaded.initialize(None)
    assert reloaded.size() == 4
    assert reloaded.index() == [0, 1, 2, 3]
    assert reloaded.output_point_vars(None) == [FV.WS, FV.WD, FV.RHO]
    reloaded_support = reloaded.get_grid_points(loaded_data=reloaded_data)
    np.testing.assert_allclose(
        np.unique(reloaded_support, axis=0),
        np.unique(_SUPPORT_POINTS, axis=0),
    )

    other_class = (
        BinnedPointCloudData if states_class is BinnedFieldData else BinnedFieldData
    )
    with pytest.raises(ValueError, match="Expected artifact class"):
        other_class(output_file).initialize(None)

    monkeypatch.undo()
    runtime_states = states_class(output_file)
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=[0.0, 0.0],
            H=90.0,
            turbine_models=["null_type"],
        ),
        verbosity=0,
    )
    algo = foxes.algorithms.Downwind(
        farm=farm,
        states=runtime_states,
        rotor_model="centre",
        wake_models=[],
        mbook=foxes.models.ModelBook(),
        verbosity=0,
    )
    with foxes.Engine.new("single", progress_bar=False, verbosity=0):
        farm_results = algo.calc_farm()

    np.testing.assert_array_equal(farm_results[FC.STATE], [0, 1, 2, 3])
    np.testing.assert_allclose(farm_results[FV.WEIGHT][:, 0], [1.0, 2.0, 1.0, 2.0])
    np.testing.assert_allclose(farm_results[FV.AMB_REWS][:, 0], [2.5, 2.5, 7.5, 7.5])
    np.testing.assert_allclose(
        farm_results[FV.AMB_WD][:, 0], [90.0, 270.0, 90.0, 270.0]
    )


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_stashes_dataset_without_worker_copy(states_class):
    artifact = _meso_binned_artifact(states_class)
    states = states_class(artifact)
    loaded_data = states.initialize(None)
    runtime_data = states.data_source
    data_stash = {}

    assert states._binned.input_source is runtime_data
    states.set_running(None, data_stash)

    assert data_stash[states.name]["data_source"] is runtime_data
    assert data_stash[states.name]["binned"]["input_source"] is runtime_data
    assert "_DatasetStates__data_source" not in vars(states)
    assert states._binned.input_source is None
    assert not any(
        isinstance(value, xr.Dataset) for value in vars(states._binned).values()
    )
    assert loaded_data["extra_data"][states.META]["data_keys"]
    with pytest.raises(ValueError, match="Cannot call set_running while running"):
        states.set_running(None, data_stash)

    states.unset_running(None, data_stash)

    assert states._binned.input_source is runtime_data
    assert "binned" not in data_stash[states.name]
    with pytest.raises(ValueError, match="Cannot call unset_running when not running"):
        states.unset_running(None, data_stash)
    states.initialize(None, loaded_data=loaded_data, force=True)

    assert states._binned.input_source is states.data_source


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_moves_source_support_to_loaded_data(
    monkeypatch,
    states_class,
):
    source = _SourceStates()
    states = _source_binned(
        states_class,
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0]},
        mean_vars=[],
    )
    loaded_data = source.initialize(None)
    _patch_source_evaluation(
        monkeypatch,
        _source_results(len(_SUPPORT_POINTS), variables=(FV.WS,)),
    )
    states.load_data(_Algorithm(), loaded_data)
    support_key = states.var("source_support")
    source_support = (
        states.support_grid
        if states_class is BinnedFieldData
        else states.support_points
    )

    assert loaded_data["extra_data"][support_key] is source_support
    data_stash = {}
    states.set_running(_Algorithm(), data_stash)

    assert states._binned.support_grid is None
    assert states._binned.support_points is None
    assert support_key in loaded_data["extra_data"]
    support_name = (
        "support_grid" if states_class is BinnedFieldData else "support_points"
    )
    assert data_stash[states.name]["binned"][support_name] is source_support

    states.unset_running(_Algorithm(), data_stash)

    restored_support = (
        states.support_grid
        if states_class is BinnedFieldData
        else states.support_points
    )
    if states_class is BinnedFieldData:
        for variable in (FV.X, FV.Y, FV.H):
            assert restored_support[variable] is source_support[variable]
    else:
        assert restored_support is source_support
    assert "binned" not in data_stash[states.name]


@pytest.mark.parametrize("class_name", [None, "UnsupportedBinnedData"])
def test_read_binned_data_rejects_invalid_class_metadata(tmp_path, class_name):
    attrs = {} if class_name is None else {"foxes_state_class": class_name}
    output_file = tmp_path / "invalid.nc"
    xr.Dataset(attrs=attrs).to_netcdf(output_file, engine=config.nc_engine)

    with pytest.raises(ValueError, match="unsupported foxes_state_class"):
        read_binned_data(output_file)


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_explicit_mean_vars_can_be_empty(states_class):
    states = _source_binned(
        states_class,
        _SourceStates(),
        bin_vars={FV.WS: [0.0, 10.0]},
        mean_vars=[],
    )

    assert states.output_point_vars(None) == [FV.WS]


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_rejects_mean_and_bin_variable_overlap(states_class):
    with pytest.raises(ValueError, match="mean_vars must not duplicate bin_vars"):
        _source_binned(
            states_class,
            _SourceStates(),
            bin_vars={FV.WS: [0.0, 10.0]},
            mean_vars=[FV.WS],
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_binned_point_cloud_rejects_invalid_nan_threshold(threshold):
    with pytest.raises(ValueError, match="nan_threshold must be between 0 and 1"):
        _source_binned(
            BinnedPointCloudData,
            _SourceStates(),
            bin_vars={FV.WS: [0.0, 10.0]},
            nan_threshold=threshold,
        )


def test_binned_data_rejects_invalid_nan_policies():
    with pytest.raises(ValueError, match="nan_policy must be"):
        _source_binned(
            BinnedPointCloudData,
            _SourceStates(),
            bin_vars={FV.WS: [0.0, 10.0]},
            nan_policy="ignore",
        )

    with pytest.raises(ValueError, match="incompatible with regular-grid topology"):
        _source_binned(
            BinnedFieldData,
            _SourceStates(),
            bin_vars={FV.WS: [0.0, 10.0]},
            nan_policy="remove",
        )


def _point_cloud_nan_artifact():
    writer = BinnedPointCloudData(
        _SourceStates(),
        bin_vars={FV.WS: [0.0, 5.0, 10.0, 15.0, 20.0]},
        mean_vars=[],
        support_points=_SUPPORT_POINTS[:3],
        write_mean_std=True,
    )
    writer._binned.calculation_vars(None)
    support = writer.support_points
    values = np.ones((4, 3))
    stats = {
        FV.WS: {name: values.copy() for name in ("mean", "std")},
    }
    data = writer._binned._create_output_dataset(
        support,
        None,
        stats,
        np.ones_like(values),
    )
    invalid_values = np.array(
        [
            [1.0, 1.0, 1.0],
            [np.nan, 2.0, 2.0],
            [np.nan, 3.0, 3.0],
            [np.nan, np.nan, 4.0],
        ]
    )
    for stat in ("mean", "std"):
        data[f"{FV.WS}_{stat}"][:] = invalid_values
    return data


def _single_bin_artifact(states_class):
    writer = _source_binned(
        states_class,
        _SourceStates(),
        bin_vars={FV.WS: [0.0, 10.0]},
        mean_vars=[],
        write_mean_std=True,
    )
    writer._binned.calculation_vars(None)
    support, axes = writer._binned._materialize_support()
    values = np.ones((1, len(support)))
    stats = {
        FV.WS: {name: values.copy() for name in ("mean", "std")},
    }
    data = writer._binned._create_output_dataset(
        support,
        axes,
        stats,
        np.ones_like(values),
    )
    return writer, support, axes, stats, data


def test_binned_point_cloud_nan_policy_raises_for_active_statistics():
    with pytest.raises(ValueError, match="Non-finite WS_mean for active bin"):
        BinnedPointCloudData(
            _point_cloud_nan_artifact(), write_mean_std=True
        ).initialize(None)


def test_binned_point_cloud_nan_policy_removes_and_interpolates():
    states = BinnedPointCloudData(
        _point_cloud_nan_artifact(),
        write_mean_std=True,
        nan_policy="remove",
        nan_threshold=0.5,
    )

    loaded_data = states.initialize(None)

    np.testing.assert_allclose(
        states.get_grid_points(loaded_data=loaded_data),
        _SUPPORT_POINTS[1:3],
    )
    assert states.size() == 4
    for data_key in loaded_data["extra_data"][states.META]["data_keys"]:
        assert np.all(np.isfinite(loaded_data["data_vars"][data_key][1]))


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_nan_policy_interpolates_active_statistics(states_class):
    _, _, _, _, data = _single_bin_artifact(states_class)
    for stat in ("mean", "std"):
        values = data[f"{FV.WS}_{stat}"].values
        values.reshape(1, -1)[0, 0] = np.nan
    states = states_class(data, write_mean_std=True, nan_policy="interpolate")

    loaded_data = states.initialize(None)

    for data_key in loaded_data["extra_data"][states.META]["data_keys"]:
        assert np.all(np.isfinite(loaded_data["data_vars"][data_key][1]))


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_writer_rejects_non_finite_weights(states_class):
    writer, support, axes, stats, _ = _single_bin_artifact(states_class)
    weights = np.ones((1, len(support)))
    weights[0, 0] = np.nan

    with pytest.raises(ValueError, match="Non-finite weights"):
        writer._binned._create_output_dataset(
            support,
            axes,
            stats,
            weights,
        )


@pytest.mark.parametrize(
    ("states_class", "invalid", "message"),
    [
        (BinnedFieldData, "weight", "Non-finite artifact weights"),
        (BinnedPointCloudData, "weight", "Non-finite artifact weights"),
        (BinnedFieldData, "bin-index", "Invalid sparse bin indices"),
        (BinnedPointCloudData, "bin-index", "Invalid sparse bin indices"),
        (BinnedFieldData, "support", "Artifact grid axes must be finite"),
        (BinnedPointCloudData, "support", "Artifact support must contain only finite"),
    ],
    ids=[
        "field-weight",
        "point-cloud-weight",
        "field-bin-index",
        "point-cloud-bin-index",
        "field-support",
        "point-cloud-support",
    ],
)
def test_binned_data_reader_rejects_non_finite_artifact_values(
    states_class,
    invalid,
    message,
):
    _, _, _, _, data = _single_bin_artifact(states_class)
    if invalid == "weight":
        data[FV.WEIGHT].values.reshape(1, -1)[0, 0] = np.nan
    elif invalid == "bin-index":
        data = data.assign_coords({FC.STATE: [1]})
    elif states_class is BinnedFieldData:
        data = data.assign_coords({FV.X: [np.nan, 500.0]})
    else:
        data[FV.X].values[0] = np.nan

    with pytest.raises(ValueError, match=message):
        states_class(data, write_mean_std=True).initialize(None)


def test_binned_data_preserves_sparse_flat_bin_indices():
    writer = BinnedPointCloudData(
        _SourceStates(),
        bin_vars={
            FV.WS: [0.0, 5.0, 10.0],
            FV.WD: [0.0, 180.0, 360.0],
        },
        mean_vars=[],
        support_points=_SUPPORT_POINTS[:2],
    )
    writer._binned.calculation_vars(None)
    weights = np.array([[0.0, 0.0], [0.2, 0.8], [0.8, 0.2], [0.0, 0.0]])
    values = np.where(weights != 0.0, 1.0, np.nan)
    stats = {
        variable: {stat: values.copy() for stat in ("mean", "std")}
        for variable in (FV.WS, FV.WD)
    }

    data = writer._binned._create_output_dataset(
        writer.support_points,
        None,
        stats,
        weights,
    )
    states = BinnedPointCloudData(data)
    loaded_data = states.initialize(None)

    np.testing.assert_array_equal(data[FC.STATE], [1, 2])
    assert FV.WS not in data
    assert "bin_centres" not in data
    assert states.index() == [1, 2]
    np.testing.assert_allclose(
        states._binned._loaded_variable(states, loaded_data, FV.WS),
        [2.5, 7.5],
    )
    np.testing.assert_allclose(
        states._binned._loaded_variable(states, loaded_data, FV.WD),
        [270.0, 90.0],
    )


def test_binned_wind_direction_wraps_at_north(monkeypatch, tmp_path):
    source = _SourceStates()
    output_file = tmp_path / "north.nc"
    states = BinnedPointCloudData(
        source,
        bin_vars={FV.WD: [350.0, 370.0]},
        mean_vars=[],
        support_points=_SUPPORT_POINTS[:1],
        interpolation="nearest",
        output_file=output_file,
        write_mean_std=True,
    )
    loaded_data = source.initialize(None)
    source_results = xr.Dataset(
        data_vars={FV.WD: ((FC.STATE, FC.POINT), [[359.0], [0.0], [1.0], [180.0]])}
    )
    _patch_source_evaluation(monkeypatch, source_results)

    states.load_data(_Algorithm(), loaded_data)

    with xr.open_dataset(output_file, engine=config.nc_engine) as data:
        np.testing.assert_array_equal(data[FC.STATE], [0])
        np.testing.assert_allclose(data[FV.WEIGHT][:, 0], [4.0])
        np.testing.assert_allclose(data[f"{FV.WD}_mean"][:, 0], [0.0], atol=1e-12)
        resultant = 0.5 * (1.0 + np.cos(np.deg2rad(1.0)))
        expected_std = np.rad2deg(np.sqrt(-2.0 * np.log(resultant)))
        np.testing.assert_allclose(data[f"{FV.WD}_std"][:, 0], [expected_std])
        assert FV.WD not in data
        assert f"{FV.WD}_min" not in data
        assert f"{FV.WD}_max" not in data


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_data_averages_non_histogram_wd_across_points(states_class):
    writer = _source_binned(
        states_class,
        _SourceStates(),
        bin_vars={FV.WS: [0.0, 20.0]},
        mean_vars=[FV.WD],
    )
    writer._binned.calculation_vars(None)
    support, axes = writer._binned._materialize_support()
    speeds = np.full((1, len(support)), 10.0)
    directions = np.where(support[:, 0] == 0.0, 359.0, 1.0)[None, :]
    artifact = writer._binned._create_output_dataset(
        support,
        axes,
        {
            FV.WS: {"mean": speeds.copy(), "std": np.zeros_like(speeds)},
            FV.WD: {"mean": directions, "std": np.zeros_like(directions)},
        },
        np.ones_like(speeds),
    )
    states = states_class(artifact)
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=[250.0, 250.0],
            H=90.0,
            turbine_models=["null_type"],
        ),
        verbosity=0,
    )
    algo = foxes.algorithms.Downwind(
        farm=farm,
        states=states,
        rotor_model="centre",
        wake_models=[],
        mbook=foxes.models.ModelBook(),
        verbosity=0,
    )

    with foxes.Engine.new("single", progress_bar=False, verbosity=0):
        farm_results = algo.calc_farm()

    direction = float(farm_results[FV.AMB_WD].item())
    angular_error = abs((direction + 180.0) % 360.0 - 180.0)
    assert angular_error < 1e-12
    np.testing.assert_allclose(
        farm_results[FV.AMB_REWS].item(),
        10.0,
    )


@pytest.mark.parametrize(
    ("states_class", "support_dims"),
    [
        (BinnedFieldData, (FV.X, FV.Y, FV.H)),
        (BinnedPointCloudData, (FC.POINT,)),
    ],
    ids=["field", "point-cloud"],
)
def test_binned_data_writes_global_means_and_optional_point_statistics(
    monkeypatch,
    tmp_path,
    states_class,
    support_dims,
):
    source = _SourceStates()
    output_file = tmp_path / f"{states_class.__name__}.nc"
    states = _source_binned(
        states_class,
        source,
        bin_vars={FV.WS: [0.0, 5.0, 10.0]},
        mean_vars=[FV.RHO],
        output_file=output_file,
        write_mean_std=True,
    )
    source_weights = np.array([1.0, 2.0, 1.0, 2.0])
    ws = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [8.0, 2.0, 8.0, 2.0],
            [8.0, 8.0, 2.0, 2.0],
            [8.0, 8.0, 8.0, 2.0],
        ]
    )
    rho = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
            [1000.0, 2000.0, 3000.0, 4000.0],
        ]
    )
    _patch_source_evaluation(
        monkeypatch,
        xr.Dataset(
            data_vars={
                FV.WS: ((FC.STATE, FC.POINT), ws),
                FV.RHO: ((FC.STATE, FC.POINT), rho),
            }
        ),
    )

    states.load_data(_Algorithm(), source.initialize(None))

    expected_weights = []
    expected_ws_mean = []
    expected_ws_std = []
    expected_rho_mean = []
    expected_rho_std = []
    expected_rho = []
    for lower, upper in zip([0.0, 5.0], [5.0, 10.0]):
        membership = (ws >= lower) & (ws < upper)
        sample_weights = source_weights[:, None] * membership
        point_weights = np.sum(sample_weights, axis=0)
        expected_weights.append(point_weights)
        for values, means, deviations in (
            (ws, expected_ws_mean, expected_ws_std),
            (rho, expected_rho_mean, expected_rho_std),
        ):
            point_mean = np.divide(
                np.sum(sample_weights * values, axis=0),
                point_weights,
                out=np.zeros_like(point_weights),
                where=point_weights > 0.0,
            )
            point_second = np.divide(
                np.sum(sample_weights * values**2, axis=0),
                point_weights,
                out=np.zeros_like(point_weights),
                where=point_weights > 0.0,
            )
            means.append(point_mean)
            deviations.append(np.sqrt(np.maximum(point_second - point_mean**2, 0.0)))
        expected_rho.append(np.sum(sample_weights * rho) / np.sum(sample_weights))

    with xr.open_dataset(output_file, engine=config.nc_engine) as data:
        assert data[FV.RHO].dims == (FC.STATE,)
        assert data[FV.WEIGHT].dims == (FC.STATE, *support_dims)
        assert FV.WS not in data
        assert "bin_centres" not in data
        np.testing.assert_allclose(data[FV.RHO], expected_rho)
        for variable in (FV.WS, FV.RHO):
            for statistic in ("mean", "std"):
                assert data[f"{variable}_{statistic}"].dims == (
                    FC.STATE,
                    *support_dims,
                )
        np.testing.assert_allclose(
            data[FV.WEIGHT].to_numpy().reshape(2, -1), expected_weights
        )
        populated = np.asarray(expected_weights) > 0.0
        for actual, expected in (
            (data[f"{FV.WS}_mean"], expected_ws_mean),
            (data[f"{FV.WS}_std"], expected_ws_std),
            (data[f"{FV.RHO}_mean"], expected_rho_mean),
            (data[f"{FV.RHO}_std"], expected_rho_std),
        ):
            actual_values = actual.to_numpy().reshape(2, -1)
            assert np.all(np.isfinite(actual_values))
            np.testing.assert_allclose(
                actual_values[populated], np.asarray(expected)[populated]
            )


def test_binned_data_rejects_point_dependent_source_weights(monkeypatch):
    source = _SourceStates()
    states = BinnedPointCloudData(
        source,
        bin_vars={FV.WS: [0.0, 10.0]},
        mean_vars=[],
        support_points=_SUPPORT_POINTS[:1],
    )
    loaded_data = source.initialize(None)
    loaded_data["data_vars"][FV.WEIGHT] = (
        (FC.STATE, FC.POINT),
        np.ones((4, 1)),
    )
    _patch_source_evaluation(
        monkeypatch,
        _source_results(1, variables=(FV.WS,)),
    )

    with pytest.raises(ValueError, match="state-dependent only"):
        states.load_data(_Algorithm(), loaded_data)


@pytest.mark.parametrize(
    "states_class",
    [BinnedFieldData, BinnedPointCloudData],
    ids=["field", "point-cloud"],
)
def test_binned_artifact_path_is_loaded_lazily(tmp_path, states_class):
    input_file = tmp_path / "missing.nc"

    states = states_class(input_file)

    assert states.data_source == input_file
    assert states.size() == 1
