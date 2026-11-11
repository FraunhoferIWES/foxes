import numpy as np
import pytest
import pickle
from types import SimpleNamespace

import foxes.constants as FC
from foxes.core import FData, MData, TData


@pytest.mark.parametrize("reverse", [False, True])
def test_constructor_broadcasts_known_dimensions_independent_of_entry_order(reverse):
    values = np.array([[3.0, 4.0, 5.0]])
    entries = [
        ("value", values),
        (FC.STATE, np.arange(4)),
        (FC.TURBINE, np.arange(3)),
    ]
    if reverse:
        entries.reverse()
    data = FData(
        data=dict(entries),
        dims={
            "value": (FC.STATE, FC.TURBINE),
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
        },
    )

    assert data["value"].shape == (4, 3)
    assert data["value"].strides[0] == 0
    assert not data["value"].flags.writeable
    assert np.shares_memory(data["value"], values)
    np.testing.assert_array_equal(data["value"], np.tile(values, (4, 1)))


def test_add_assignment_and_update_expand_to_declared_sizes():
    data = FData.from_sizes(4, 3)
    source = np.array([[2.0]])
    data.add("value", source, (FC.STATE, FC.TURBINE))
    assert data["value"].shape == (4, 3)
    assert np.shares_memory(source, data["value"])

    for replace in (
        lambda: data.__setitem__("value", np.array([[5.0]])),
        lambda: data.update({"value": np.array([[7.0]])}),
    ):
        replace()
        assert data["value"].shape == (4, 3)
        assert not data["value"].flags.writeable

    full = np.ones((4, 3))
    data["value"] = full
    assert data["value"] is full
    assert data["value"].flags.writeable


def test_discovering_larger_dimension_expands_existing_entries():
    data = MData(data={"value": np.array([3.0])}, dims={"value": (FC.STATE,)})
    data.add(FC.STATE, np.arange(4), (FC.STATE,))

    assert data["value"].shape == (4,)
    assert not data["value"].flags.writeable
    np.testing.assert_array_equal(data["value"], [3.0] * 4)


def test_broadcast_data_slices_preserve_values_and_storage():
    source = np.array([[2.0, 3.0, 4.0]])
    data = FData.from_sizes(4, 3)
    data.add("value", source, (FC.STATE, FC.TURBINE))

    subset = data.get_slice(FC.STATE, slice(1, 3))

    assert subset["value"].shape == (2, 3)
    assert np.shares_memory(subset["value"], source)
    np.testing.assert_array_equal(subset["value"], [[2, 3, 4]] * 2)


def test_serialization_keeps_broadcast_storage_bounded():
    data = FData.from_sizes(10_000, 3)
    data.add("value", np.array([[2.0, 3.0, 4.0]]), (FC.STATE, FC.TURBINE))

    encoded = pickle.dumps(data)
    restored = pickle.loads(encoded)
    materialized = FData.from_sizes(10_000, 3)
    materialized.add(
        "value",
        np.tile(np.array([[2.0, 3.0, 4.0]]), (10_000, 1)),
        (FC.STATE, FC.TURBINE),
    )

    assert len(encoded) < len(pickle.dumps(materialized))
    assert restored["value"].shape == (10_000, 3)
    assert restored["value"].strides[0] == 0
    assert restored.sizes == data.sizes
    assert not restored["value"].flags.writeable
    np.testing.assert_array_equal(restored["value"], data["value"])


def test_dataset_copy_keeps_broadcast_storage_independent_and_bounded():
    import xarray as xr

    source = np.array([[2.0, 3.0, 4.0]])
    dataset = xr.Dataset(
        {"value": ((FC.STATE, FC.TURBINE), np.broadcast_to(source, (4, 3)))}
    )

    data = FData.from_dataset(dataset)

    assert data["value"].shape == (4, 3)
    assert data["value"].strides[0] == 0
    assert not np.shares_memory(data["value"], source)
    np.testing.assert_array_equal(data["value"], np.tile(source, (4, 1)))


@pytest.mark.parametrize("coordinate", [FC.FARM, FC.CLUSTER])
@pytest.mark.parametrize("n_turbines", [1, 3])
def test_farm_dataset_keeps_static_memberships_when_slicing_states(
    coordinate, n_turbines
):
    import xarray as xr

    memberships = np.array(["north", None, "south"], dtype=object)[:n_turbines]
    dataset = xr.Dataset(
        {"value": ((FC.STATE, FC.TURBINE), np.ones((4, n_turbines)))},
        coords={
            FC.STATE: np.arange(4),
            FC.TURBINE: np.arange(n_turbines),
            coordinate: (FC.TURBINE, memberships),
        },
    )
    data = FData.from_dataset(dataset, s_states=slice(1, 3))

    assert data["value"].shape == (2, n_turbines)
    assert data.dims[coordinate] == (FC.TURBINE,)
    np.testing.assert_array_equal(data[coordinate], memberships)
    assert not np.shares_memory(data[coordinate], memberships)


@pytest.mark.parametrize("coordinate", [FC.FARM, FC.CLUSTER])
def test_farm_memberships_do_not_accept_state_only_fields(coordinate):
    with pytest.raises(ValueError, match=f"Invalid shape for '{coordinate}'"):
        FData(
            data={
                FC.STATE: np.arange(2),
                FC.TURBINE: np.arange(3),
                coordinate: np.array(["north", "south"]),
            },
            dims={
                FC.STATE: (FC.STATE,),
                FC.TURBINE: (FC.TURBINE,),
                coordinate: (FC.STATE,),
            },
        )


def test_target_constructor_expands_coordinates_and_fields():
    points = np.zeros((1, 3, 2, 3))
    data = TData.from_tpoints(
        points,
        np.array([0.25, 0.75]),
        data={FC.STATE: np.arange(4), "value": np.array([[[8.0]]])},
        dims={FC.STATE: (FC.STATE,), "value": (FC.STATE, FC.TARGET, FC.TPOINT)},
    )

    assert data[FC.TARGETS].shape == (4, 3, 2, 3)
    assert np.shares_memory(data[FC.TARGETS], points)
    assert data["value"].shape == (4, 3, 2)
    subset = data.get_targets_subset([2, 0])
    assert subset[FC.TARGETS].shape == (4, 2, 2, 3)
    assert subset["value"].shape == (4, 2, 2)
    np.testing.assert_array_equal(subset["value"], np.full((4, 2, 2), 8.0))


def test_genuinely_single_sized_dimensions_are_unchanged():
    source = np.ones((1, 1))
    data = FData(data={"value": source}, dims={"value": (FC.STATE, FC.TURBINE)})
    assert data["value"] is source
    assert data["value"].flags.writeable


def test_incompatible_replacements_raise_contextual_errors():
    data = FData.from_sizes(4, 3)
    data.add("value", np.zeros((4, 3)), (FC.STATE, FC.TURBINE))

    with pytest.raises(ValueError, match="value.*shape.*"):
        data["value"] = np.ones((2, 3))
    with pytest.raises(ValueError, match="value.*dimensions"):
        data["value"] = np.ones(4)
    assert data["value"].shape == (4, 3)


def test_sequential_wake_lookup_does_not_mutate_previous_results():
    import xarray as xr
    from foxes.models.wake_frames.seq_dynamic_wakes import SeqDynamicWakes

    previous = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    algo = SimpleNamespace(
        states=SimpleNamespace(counter=2),
        farm_results_downwind=xr.Dataset(
            {"value": ((FC.STATE, FC.TURBINE), previous.copy())}
        ),
    )
    fdata = FData(
        data={"value": np.array([[90.0, 91.0]])},
        dims={"value": (FC.STATE, FC.TURBINE)},
    )
    tdata = TData.from_tpoints(np.zeros((1, 2, 2, 3)), np.array([0.25, 0.75]))
    tdata.add(
        FC.STATES_SEL, np.array([[[0, 2], [1, 2]]]), (FC.STATE, FC.TARGET, FC.TPOINT)
    )
    tdata[FC.STATE_SOURCE_ORDERI] = 0
    frame = SeqDynamicWakes(dt_min=10.0)

    values = frame.get_wake_modelling_data(
        algo, "value", 0, fdata, tdata, FC.STATE_TARGET_TPOINT
    )
    mean = frame.get_wake_modelling_data(
        algo, "value", 0, fdata, tdata, FC.STATE_TARGET
    )

    np.testing.assert_array_equal(values, [[[10.0, 90.0], [20.0, 90.0]]])
    np.testing.assert_array_equal(mean, [[70.0, 72.5]])
    np.testing.assert_array_equal(algo.farm_results_downwind["value"], previous)
