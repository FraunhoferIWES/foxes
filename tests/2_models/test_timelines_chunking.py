"""Timeline source histories across state-chunk boundaries."""

import numpy as np
import pandas as pd
import pytest

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.iterative.models.convergence import ConvVarDelta


def _calculate(engine, chunk_size):
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {
                FV.WS: [7.0, 8.0, 9.0, 8.5, 7.5, 8.0],
                FV.WD: [270.0, 272.0, 274.0, 271.0, 269.0, 270.0],
            },
            index=pd.date_range("2020-01-01", periods=6, freq="min"),
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.TI: 0.07, FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for x in (0.0, 600.0, 1200.0):
        farm.add_turbine(
            foxes.Turbine(xy=[x, 0.0], turbine_models=["NREL5MW"]), verbosity=0
        )
    algo = foxes.algorithms.Iterative(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear_lim_k004"],
        wake_frame="timelines",
        max_wake_length_km=8,
        conv_crit=ConvVarDelta({FV.REWS: 1e-8}),
        verbosity=0,
    )
    with foxes.Engine.new(
        engine,
        n_procs=2,
        chunk_size_states=chunk_size,
        chunk_size_points=4,
        verbosity=0,
    ):
        return algo.calc_farm(outputs="farm_results_all")


@pytest.mark.parametrize("engine", ["numpy", "process"])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_timeline_chunked_history_matches_full_states(engine, chunk_size):
    # First-iteration source guesses must converge to full-history results.
    expected = _calculate("single", 6)
    actual = _calculate(engine, chunk_size)
    np.testing.assert_array_equal(actual[FC.STATE], expected[FC.STATE])
    for variable in (FV.REWS, FV.CT, FV.P):
        assert np.isfinite(actual[variable]).all()
        np.testing.assert_allclose(actual[variable], expected[variable], rtol=1e-7)
