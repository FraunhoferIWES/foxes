import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace

import foxes.constants as FC
from foxes.core import FData, MData, Model, TData, WakeFrame


class _ModelWithData(Model):
    def __init__(self):
        super().__init__()
        self.n_states = 2
        self.n_targets = 3
        self.value = np.array([1.0, 2.0])


class _WakeFrame(WakeFrame):
    def calc_order(self, algo, mdata, fdata):
        raise NotImplementedError

    def get_wake_coos(self, algo, mdata, fdata, tdata, downwind_index):
        raise NotImplementedError


def test_get_data_boolean_selection_broadcasts_before_indexing():
    model = _ModelWithData()
    selection = np.array([[True, False, True], [False, True, False]])

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        selection=selection,
    )

    assert np.array_equal(out, np.array([1.0, 1.0, 2.0]))
    assert out.flags.writeable


@pytest.mark.parametrize("full_shape", [False, True])
def test_get_data_expands_only_mismatched_shapes(full_shape):
    model = _ModelWithData()
    if full_shape:
        model.value = np.arange(6.0).reshape(2, 3)

    out = model.get_data("value", FC.STATE_TARGET, lookup="s")

    expected = model.value if full_shape else np.array([[1.0] * 3, [2.0] * 3])
    np.testing.assert_array_equal(out, expected)
    assert out.flags.writeable == full_shape
    assert np.shares_memory(out, model.value)
    assert model.value.flags.writeable
    if not full_shape:
        with pytest.raises(ValueError, match="read-only"):
            out[0, 0] = 9.0


@pytest.mark.parametrize("value", [np.ones((2, 4)), np.ones((3,)), np.ones((2, 3, 1))])
def test_get_data_rejects_incompatible_shapes(value):
    model = _ModelWithData()
    model.value = value

    with pytest.raises(ValueError, match="value.*shape|Shape mismatch.*value"):
        model.get_data("value", FC.STATE_TARGET, lookup="s")


@pytest.mark.parametrize(
    "selection",
    [
        slice(None),
        slice(1, None),
        (slice(None), slice(1, None)),
        (1, slice(None)),
        (slice(None), 1),
        (1, 2),
        (Ellipsis, 1),
        [slice(None), slice(1, None)],
        np.array([True, False]),
        np.array([[True, False, True], [False, True, False]]),
        np.zeros((2, 3), dtype=bool),
    ],
)
def test_get_data_selection_matches_numpy(selection):
    model = _ModelWithData()
    index = tuple(selection) if isinstance(selection, list) else selection
    expected = np.broadcast_to(model.value[:, None], (2, 3))[index]

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        selection=selection,
    )

    np.testing.assert_array_equal(out, expected)
    assert out.shape == expected.shape


def test_get_data_default_lookup_includes_target_data():
    tdata = TData.from_points(np.zeros((2, 3, 3)))
    tdata.add(
        "value",
        np.arange(6.0).reshape(2, 3, 1),
        (FC.STATE, FC.TARGET, FC.TPOINT),
    )

    out = Model().get_data("value", FC.STATE_TARGET_TPOINT, tdata=tdata)

    np.testing.assert_array_equal(out, tdata["value"])
    assert out.flags.writeable
    assert np.shares_memory(out, tdata["value"])


def test_get_data_always_expands_to_target_shape():
    model = _ModelWithData()

    out = model.get_data("value", FC.STATE_TARGET)

    assert out.shape == (2, 3)
    assert not out.flags.writeable
    assert np.shares_memory(out, model.value)


@pytest.mark.parametrize("selection", [3, "invalid", np.array([0, 1])])
def test_get_data_rejects_unsupported_selection(selection):
    with pytest.raises(TypeError, match="selection"):
        _ModelWithData().get_data("value", FC.STATE_TARGET, selection=selection)


def test_get_data_validates_target_and_dimension_metadata():
    with pytest.raises(KeyError, match="target"):
        Model().get_data("value", "invalid")
    with pytest.raises(ValueError, match="downwind_index"):
        Model().get_data("value", FC.STATE_TURBINE, downwind_index=0)

    model = Model()
    model.value = 1.0
    with pytest.raises(KeyError, match="n_states"):
        model.get_data("value", FC.STATE_TARGET)


def test_get_data_missing_optional_value_needs_no_sizes():
    assert Model().get_data("missing", FC.STATE_TARGET, accept_none=True) is None
    with pytest.raises(ValueError, match="missing.*not found"):
        Model().get_data("missing", FC.STATE_TARGET)


@pytest.mark.parametrize("lookup", ["p", "smfp", "invalid"])
def test_get_data_rejects_unknown_lookup_sources(lookup):
    with pytest.raises(ValueError, match="lookup"):
        _ModelWithData().get_data("value", FC.STATE_TARGET, lookup=lookup)


@pytest.mark.parametrize(
    ("dims", "values"),
    [
        ((FC.TARGET,), np.array([1.0, 2.0, 3.0])),
        ((FC.TARGET, FC.STATE), np.arange(6.0).reshape(3, 2)),
        ((), np.array(4.0)),
    ],
)
def test_get_data_aligns_named_dimensions(dims, values):
    model = _ModelWithData()
    mdata = MData(data={"value": values}, dims={"value": dims})
    expected = values.T if len(dims) == 2 else values
    expected = np.broadcast_to(expected, (2, 3))

    out = model.get_data("value", FC.STATE_TARGET, lookup="m", mdata=mdata)

    np.testing.assert_array_equal(out, expected)


def test_get_data_farm_source_selects_downwind_turbine():
    fdata = FData.from_sizes(2, 4)
    fdata.add("value", np.arange(8.0).reshape(2, 4), (FC.STATE, FC.TURBINE))
    tdata = TData.from_points(np.zeros((2, 3, 3)))

    out = Model().get_data(
        "value",
        FC.STATE_TARGET_TPOINT,
        lookup="f",
        fdata=fdata,
        tdata=tdata,
        downwind_index=2,
    )

    np.testing.assert_array_equal(out[:, :, 0], np.array([[2.0] * 3, [6.0] * 3]))
    assert not out.flags.writeable


def test_get_data_nan_validation_checks_selected_values():
    model = _ModelWithData()
    model.value = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(ValueError, match="value.*NaN"):
        model.get_data("value", FC.STATE_TARGET, lookup="s", accept_nan=False)

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        accept_nan=False,
        selection=(slice(None), 0),
    )
    np.testing.assert_array_equal(out, [1.0, 4.0])


def test_get_data_nan_validation_accepts_empty_and_nonnumeric_arrays():
    model = _ModelWithData()
    model.value = np.array(["a", "b"])
    out = model.get_data("value", FC.STATE_TARGET, lookup="s", accept_nan=False)
    np.testing.assert_array_equal(out[:, 0], ["a", "b"])

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        accept_nan=False,
        selection=np.zeros((2, 3), dtype=bool),
    )
    assert out.shape == (0,)


def _iteration_data():
    fdata = FData(
        data={"value": np.array([[30.0], [40.0]])},
        dims={"value": (FC.STATE, FC.TURBINE)},
        states_i0=2,
    )
    tdata = TData.from_points(np.zeros((2, 3, 3)))
    tdata.add(
        FC.STATES_SEL,
        np.array([[[0], [1], [2]], [[1], [2], [3]]]),
        (FC.STATE, FC.TARGET, FC.TPOINT),
    )
    tdata[FC.STATE_SOURCE_ORDERI] = 0
    algo = SimpleNamespace(
        wake_frame=_WakeFrame(),
        farm_results_downwind=xr.Dataset(
            {"value": ((FC.STATE, FC.TURBINE), np.array([[10.0], [20.0]]))}
        ),
    )
    return fdata, tdata, algo


@pytest.mark.parametrize("target", [FC.STATE_TARGET, FC.STATE_TARGET_TPOINT])
def test_get_data_iteration_preserves_sources(target):
    fdata, tdata, algo = _iteration_data()
    original = fdata["value"].copy()

    out = Model().get_data(
        "value",
        target,
        lookup="f",
        fdata=fdata,
        tdata=tdata,
        algo=algo,
        downwind_index=0,
    )

    np.testing.assert_array_equal(out.reshape(2, 3), [[10, 20, 30], [20, 30, 40]])
    np.testing.assert_array_equal(fdata["value"], original)
    assert out.flags.writeable


def test_get_data_iteration_does_not_override_model_lookup():
    fdata, tdata, algo = _iteration_data()
    model = _ModelWithData()

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="sf",
        fdata=fdata,
        tdata=tdata,
        algo=algo,
    )

    np.testing.assert_array_equal(out, [[1.0] * 3, [2.0] * 3])


def test_get_data_iteration_requires_previous_results():
    fdata, tdata, algo = _iteration_data()
    algo.farm_results_downwind = None

    with pytest.raises(ValueError, match="previous iteration"):
        Model().get_data(
            "value",
            FC.STATE_TARGET,
            lookup="f",
            fdata=fdata,
            tdata=tdata,
            algo=algo,
            downwind_index=0,
        )


def test_get_data_three_dimensional_selection_preserves_trailing_points():
    tdata = TData.from_tpoints(np.zeros((2, 3, 4, 3)), np.full(4, 0.25))
    model = _ModelWithData()
    mask = np.array([[True, False, True], [False, True, False]])

    out = model.get_data(
        "value",
        FC.STATE_TARGET_TPOINT,
        tdata=tdata,
        selection=mask,
    )

    np.testing.assert_array_equal(out, [[1.0] * 4, [1.0] * 4, [2.0] * 4])
    assert out.flags.writeable


def test_get_data_iteration_averages_target_point_state_indices():
    fdata, tdata, algo = _iteration_data()
    tdata = TData.from_tpoints(np.zeros((2, 3, 2, 3)), np.array([0.25, 0.75]))
    tdata.add(
        FC.STATES_SEL,
        np.array([[[0, 1], [1, 2], [2, 3]], [[0, 1], [1, 2], [2, 3]]]),
        (FC.STATE, FC.TARGET, FC.TPOINT),
    )
    tdata[FC.STATE_SOURCE_ORDERI] = 0

    out = Model().get_data(
        "value",
        FC.STATE_TARGET,
        lookup="f",
        fdata=fdata,
        tdata=tdata,
        algo=algo,
        downwind_index=0,
    )

    np.testing.assert_array_equal(out, [[20, 30, 40], [20, 30, 40]])
    assert out.flags.writeable


def test_get_data_tuple_selection_retains_broadcast_view():
    model = _ModelWithData()

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        selection=(slice(None), slice(None)),
    )

    assert out.shape == (2, 3)
    assert not out.flags.writeable
    assert np.shares_memory(out, model.value)


def test_wake_frame_modelling_data_broadcasts_to_target_shape():
    frame = _WakeFrame()
    fdata = FData.from_sizes(2, 4)
    fdata["value"] = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    tdata = TData.from_points(np.zeros((2, 3, 3)))

    out = frame.get_wake_modelling_data(
        None,
        "value",
        1,
        fdata,
        tdata,
        FC.STATE_TARGET_TPOINT,
    )

    assert out.shape == (2, 3, 1)
    assert np.array_equal(out[:, :, 0], np.array([[2.0] * 3, [6.0] * 3]))
    assert not out.flags.writeable
