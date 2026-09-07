import numpy as np

import foxes.constants as FC
from foxes.core import FData, Model, TData, WakeFrame


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


def test_get_data_upcast_returns_read_only_broadcast_view():
    model = _ModelWithData()

    out = model.get_data("value", FC.STATE_TARGET, lookup="s", upcast=True)

    assert out.shape == (2, 3)
    assert np.array_equal(out, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    assert not out.flags.writeable


def test_get_data_tuple_selection_keeps_materialized_result_writable():
    model = _ModelWithData()

    out = model.get_data(
        "value",
        FC.STATE_TARGET,
        lookup="s",
        selection=(slice(None), slice(None)),
    )

    assert out.shape == (2, 3)
    assert out.flags.writeable


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
