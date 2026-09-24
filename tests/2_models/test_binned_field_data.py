import numpy as np

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import States
from foxes.input.states.binned import BinnedFieldData


class _SourceStates(States):
    def size(self):
        return 1

    def output_point_vars(self, algo):
        return [FV.WS, FV.WD]

    def calculate(self, algo, mdata, fdata, tdata):
        return {}


class _ObservedBinnedFieldData(BinnedFieldData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_chunks = []

    def calculate(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        is_turbine_point_cloud=False,
    ):
        results = super().calculate(
            algo,
            mdata,
            fdata,
            tdata,
            is_turbine_point_cloud=is_turbine_point_cloud,
        )
        self.weight_chunks.append((mdata[FC.STATE].copy(), tdata[FV.WEIGHT].copy()))
        return results


def _binned_field_artifact():
    writer = BinnedFieldData(
        _SourceStates(),
        bin_vars={
            FV.WS: [0.0, 5.0, 10.0],
            FV.WD: [0.0, 180.0, 360.0],
        },
        mean_vars=[],
        support_grid={
            FV.X: [0.0, 100.0],
            FV.Y: [0.0, 100.0],
            FV.H: [80.0, 100.0],
        },
    )
    writer._binned.calculation_vars(None)
    support, axes = writer._binned._materialize_support()
    weights = np.zeros((4, len(support)))
    at_second_turbine = support[:, 0] == 100.0
    weights[1] = np.where(at_second_turbine, 0.8, 0.2)
    weights[2] = np.where(at_second_turbine, 0.2, 0.8)

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
    return writer._binned._create_output_dataset(
        support,
        axes,
        stats,
        weights,
    )


def test_binned_field_data_chunks_target_dependent_weights():
    artifact = _binned_field_artifact()

    loaded_states = BinnedFieldData(artifact)
    loaded_data = loaded_states.initialize(None)
    data_key = loaded_data["extra_data"][loaded_states.META]["data_keys"][0]
    dims, values = loaded_data["data_vars"][data_key]
    variable_names = loaded_data["coords"][dims[-1]].tolist()

    assert dims[0] == FC.STATE
    assert values.shape[0] == 2
    assert FV.WEIGHT in variable_names
    np.testing.assert_array_equal(loaded_data["coords"][FC.STATE], [1, 2])

    states = _ObservedBinnedFieldData(artifact)
    farm = foxes.WindFarm()
    for x in (0.0, 100.0):
        farm.add_turbine(
            foxes.Turbine(
                xy=[x, 0.0],
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

    with foxes.Engine.new(
        "numpy",
        chunk_size_states=1,
        progress_bar=False,
        verbosity=0,
    ):
        farm_results = algo.calc_farm()
        states.weight_chunks.clear()
        algo.calc_points(
            farm_results,
            np.array([[0.0, 0.0, 90.0], [100.0, 0.0, 90.0]]),
            outputs=[FV.AMB_WS, FV.AMB_WD],
            ambient=True,
        )

    assert len(states.weight_chunks) == 2
    np.testing.assert_array_equal(states.weight_chunks[0][0], [1])
    np.testing.assert_array_equal(states.weight_chunks[1][0], [2])
    assert states.weight_chunks[0][1].shape == (1, 2, 1)
    assert states.weight_chunks[1][1].shape == (1, 2, 1)
    np.testing.assert_allclose(states.weight_chunks[0][1][0, :, 0], [0.2, 0.8])
    np.testing.assert_allclose(states.weight_chunks[1][1][0, :, 0], [0.8, 0.2])
    np.testing.assert_array_equal(farm_results[FC.STATE], [1, 2])
    np.testing.assert_allclose(
        farm_results[FV.WEIGHT],
        [[0.2, 0.8], [0.8, 0.2]],
    )
    np.testing.assert_allclose(farm_results[FV.AMB_WD], [[270.0] * 2, [90.0] * 2])
