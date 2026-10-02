import inspect

import numpy as np
import pandas as pd

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.sequential.sequential import Sequential
from foxes.algorithms.iterative.models.convergence import ConvVarDelta


def test_sequential_does_not_override_global_n_states():
    """Sequential steps must not mutate the algorithm-global state count."""
    source = inspect.getsource(Sequential.__next__)
    assert "self.n_states = 1" not in source
    assert "self.n_states = len(self._inds)" not in source


def test_sequential_uses_state_labels_for_generated_selections():
    state_index = pd.date_range("2020-01-01", periods=3, freq="h")
    states = foxes.input.states.StatesTable(
        data_source=pd.DataFrame(
            {
                FV.WS: [7.0, 8.0, 9.0],
                FV.WD: [270.0, 270.0, 270.0],
                FV.TI: [0.06, 0.06, 0.06],
            },
            index=state_index,
        ),
        output_vars=[FV.WS, FV.WD, FV.TI],
    )
    farm = foxes.WindFarm()
    farm.add_turbine(
        foxes.Turbine(
            xy=[0.0, 0.0],
            H=100.0,
            turbine_models=["null_type"],
        ),
        verbosity=0,
    )
    points = np.array([[[0.0, 0.0, 100.0]]])
    algo = Sequential(
        farm,
        states,
        points=points,
        ambient=True,
        rotor_model="centre",
        wake_models=[],
        conv_crit=ConvVarDelta({FV.REWS: 1e-6}),
        mod_cutin={"modify_ct": False},
        verbosity=0,
    )

    with foxes.Engine.new("single", verbosity=0):
        list(algo)

    np.testing.assert_array_equal(algo.farm_results[FC.STATE], state_index)
    assert algo.point_results is not None
    np.testing.assert_array_equal(algo.point_results[FC.STATE], state_index)
    np.testing.assert_allclose(algo.point_results[FV.WS].to_numpy()[:, 0], [7, 8, 9])
