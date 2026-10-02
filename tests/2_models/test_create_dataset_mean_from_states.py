import numpy as np
import pandas as pd
import pytest
import xarray as xr

import foxes
import foxes.variables as FV
from foxes.input.states.create import create_dataset_mean_from_states


def _micro_states(
    x=(-100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0),
    y=(100.0, 150.0, 200.0, 250.0, 300.0, 350.0),
):
    data = xr.Dataset(
        coords={
            "state": [0],
            "x": np.asarray(x),
            "y": np.asarray(y),
            "height": [60.0, 80.0, 100.0, 120.0],
        }
    )
    return foxes.input.states.FieldData(
        data_source=data,
        output_vars=[FV.WS, FV.WD],
        states_coord="state",
        x_coord="x",
        y_coord="y",
        h_coord="height",
        time_format=None,
        height_bounds=(80.0, 100.0),
    )


def _boundary():
    return foxes.utils.geom2d.ClosedPolygon(
        np.array(
            [
                [0.0, 200.0],
                [100.0, 200.0],
                [100.0, 250.0],
                [0.0, 250.0],
            ]
        )
    )


def test_create_dataset_mean_from_states_is_single_state_field_compatible():
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: [8.0, 8.0],
                FV.WD: [350.0, 10.0],
                FV.TI: [0.06, 0.10],
                FV.WEIGHT: [0.25, 0.75],
            }
        ),
        output_vars=[FV.WS, FV.WD, FV.TI],
    )
    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean_from_states(
            states=states,
            micro_states=_micro_states(),
            output_vars=[FV.WS, FV.WD, FV.TI],
            boundary=_boundary(),
            add_uv=True,
            states_batch_size=1,
            points_batch_size=5,
            wd_histo_width=30.0,
            verbosity=0,
        )

    assert data.sizes == {"x": 5, "y": 4, "height": 2}
    np.testing.assert_array_equal(data.x, [-50.0, 0.0, 50.0, 100.0, 150.0])
    np.testing.assert_array_equal(data.y, [150.0, 200.0, 250.0, 300.0])
    np.testing.assert_array_equal(data.height, [80.0, 100.0])
    np.testing.assert_allclose(data[FV.WS], 7.90902161)
    np.testing.assert_allclose(data[FV.WD], 5.03836877)
    np.testing.assert_allclose(data[FV.MAIN_WD], 5.03836877)
    np.testing.assert_allclose(data[FV.TI], 0.09)
    np.testing.assert_allclose(data[FV.U], -0.69459271)
    np.testing.assert_allclose(data[FV.V], -7.87846202)

    field = foxes.input.states.SingleStateField(
        data_source=data,
        output_vars=[FV.WS, FV.WD, FV.TI],
    )
    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    field.load_data(algo=None, loaded_data=loaded_data, verbosity=0)
    assert field.size() == 1


def test_create_dataset_mean_from_states_captures_mode_across_bin_boundary():
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: [8.0, 8.0, 8.0],
                FV.WD: [4.9, 5.1, 90.0],
                FV.WEIGHT: [1.0 / 3.0] * 3,
            }
        ),
        output_vars=[FV.WS, FV.WD],
    )
    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean_from_states(
            states=states,
            micro_states=_micro_states(x=(0.0,), y=(200.0,)),
            output_vars=[FV.WS, FV.WD],
            states_batch_size=1,
            wd_histo_width=10.0,
            verbosity=0,
        )

    np.testing.assert_allclose(data[FV.MAIN_WD], 5.0)


@pytest.mark.parametrize(
    "state_index",
    [
        pd.date_range("2020-01-01", periods=201, freq="10min"),
        pd.Index([f"state-{index}" for index in range(201)]),
        pd.Index(7 + 10 * np.arange(201)),
    ],
    ids=["datetime", "string", "nonconsecutive-integer"],
)
@pytest.mark.parametrize("engine_type", ["single", "threads"])
def test_create_dataset_mean_from_states_batches_state_labels(state_index, engine_type):
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: np.full(len(state_index), 8.0),
                FV.WD: np.full(len(state_index), 270.0),
            },
            index=state_index,
        ),
        output_vars=[FV.WS, FV.WD],
    )

    with foxes.Engine.new(engine_type, n_procs=2, verbosity=0):
        data = create_dataset_mean_from_states(
            states=states,
            micro_states=_micro_states(x=(0.0,), y=(200.0,)),
            output_vars=[FV.WS, FV.WD],
            states_batch_size=100,
            points_batch_size=1,
            verbosity=0,
        )

    np.testing.assert_allclose(data[FV.WS], 8.0)
    np.testing.assert_allclose(data[FV.WD], 270.0)


def test_create_dataset_mean_from_states_rejects_duplicate_state_labels():
    state_index = pd.date_range("2020-01-01", periods=101, freq="10min").repeat(2)[:201]
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: np.full(len(state_index), 8.0),
                FV.WD: np.full(len(state_index), 270.0),
            },
            index=state_index,
        ),
        output_vars=[FV.WS, FV.WD],
    )

    with foxes.Engine.new("single", verbosity=0):
        with pytest.raises(ValueError, match="requires unique state labels"):
            create_dataset_mean_from_states(
                states=states,
                micro_states=_micro_states(x=(0.0,), y=(200.0,)),
                output_vars=[FV.WS, FV.WD],
                states_batch_size=100,
                points_batch_size=1,
                verbosity=0,
            )


def test_create_dataset_mean_from_states_uses_complete_micro_grid_without_boundary():
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame({FV.WS: [8.0], FV.WD: [270.0]}),
        output_vars=[FV.WS, FV.WD],
    )
    micro_states = _micro_states(x=(0.0, 50.0), y=(200.0, 250.0))

    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean_from_states(
            states=states,
            micro_states=micro_states,
            output_vars=[FV.WS, FV.WD],
            verbosity=0,
        )

    assert data.sizes == {"x": 2, "y": 2, "height": 2}
    np.testing.assert_array_equal(data.x, [0.0, 50.0])
    np.testing.assert_array_equal(data.y, [200.0, 250.0])
    np.testing.assert_array_equal(data.height, [80.0, 100.0])


def test_create_dataset_mean_from_states_selects_grid_for_irregular_points(
    tmp_path,
):
    coordinates = np.array(
        [
            [-200.0, -200.0],
            [-100.0, -200.0],
            [0.0, -200.0],
            [100.0, -200.0],
            [200.0, -200.0],
            [200.0, -100.0],
            [200.0, 0.0],
            [200.0, 100.0],
            [200.0, 200.0],
            [100.0, 200.0],
            [0.0, 200.0],
            [-100.0, 200.0],
            [-200.0, 200.0],
            [-200.0, 100.0],
            [-200.0, 0.0],
            [-200.0, -100.0],
        ]
    )
    source = xr.Dataset(
        data_vars={
            FV.X: (("point",), coordinates[:, 0]),
            FV.Y: (("point",), coordinates[:, 1]),
            FV.H: (("point",), np.full(len(coordinates), 100.0)),
        },
        coords={"state": [0]},
    )
    micro_states = foxes.input.states.PointCloudData(
        data_source=source,
        output_vars=[FV.WS, FV.WD],
        states_coord="state",
        point_coord="point",
        x_ncvar=FV.X,
        y_ncvar=FV.Y,
        h_ncvar=FV.H,
    )
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame({FV.WS: [8.0], FV.WD: [270.0]}),
        output_vars=[FV.WS, FV.WD],
    )
    boundary = foxes.utils.geom2d.ClosedPolygon(
        np.array(
            [
                [-50.0, -50.0],
                [50.0, -50.0],
                [50.0, 50.0],
                [-50.0, 50.0],
            ]
        )
    )
    plot_file = tmp_path / "mean_grid.png"

    with foxes.Engine.new("default", verbosity=0):
        data = create_dataset_mean_from_states(
            states=states,
            micro_states=micro_states,
            output_vars=[FV.WS, FV.WD],
            boundary=boundary,
            grid_point_plot=plot_file,
            verbosity=0,
        )

    expected_axis = [-200.0, -100.0, 0.0, 100.0, 200.0]
    assert data.sizes == {"x": 5, "y": 5, "height": 1}
    np.testing.assert_array_equal(data.x, expected_axis)
    np.testing.assert_array_equal(data.y, expected_axis)
    np.testing.assert_array_equal(data.height, [100.0])
    assert plot_file.stat().st_size > 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"output_vars": [FV.WS]}, "must be requested together"),
        ({"states_batch_size": 0}, "must be positive"),
        ({"wd_histo_width": 0.0}, "bin width must be in"),
    ],
)
def test_create_dataset_mean_from_states_rejects_invalid_input(kwargs, message):
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame({FV.WS: [8.0], FV.WD: [270.0]}),
        output_vars=[FV.WS, FV.WD],
    )
    parameters = {
        "states": states,
        "micro_states": _micro_states(),
        "output_vars": [FV.WS, FV.WD],
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        create_dataset_mean_from_states(**parameters)


@pytest.mark.parametrize(
    "micro_states, boundary, message",
    [
        (_micro_states(x=(1.0, 0.0)), None, "strictly increasing"),
        (
            _micro_states(),
            foxes.utils.geom2d.ClosedPolygon(
                np.array(
                    [
                        [-100.0, 200.0],
                        [100.0, 200.0],
                        [100.0, 250.0],
                        [-100.0, 250.0],
                    ]
                )
            ),
            "exterior points",
        ),
    ],
)
def test_create_dataset_mean_from_states_rejects_invalid_grid(
    micro_states, boundary, message
):
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame({FV.WS: [8.0], FV.WD: [270.0]}),
        output_vars=[FV.WS, FV.WD],
    )

    with pytest.raises(ValueError, match=message):
        create_dataset_mean_from_states(
            states=states,
            micro_states=micro_states,
            output_vars=[FV.WS, FV.WD],
            boundary=boundary,
        )
