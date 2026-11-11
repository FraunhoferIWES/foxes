"""Synthetic algorithms for isolated constructor-defined output tests."""

from copy import deepcopy

import numpy as np
import pandas as pd

import foxes
import foxes.constants as FC
import foxes.variables as FV


def run_output(output, farm_results=None, *, farm=None, algo=None, point_results=None):
    """
    Execute an output with supplied results and optional synthetic farm metadata.

    Parameters
    ----------
    output
        Constructor-defined output to run through ``calc_outputs``.
    farm_results
        Optional precomputed farm dataset. Its state and turbine coordinates
        determine the synthetic algorithm's dimensions when needed.
    farm
        Optional farm for a synthetic algorithm. A private copy receives any
        missing null turbine models, leaving the caller's farm unchanged.
    algo
        Existing algorithm to use instead of creating a synthetic one.
    point_results
        Optional precomputed point dataset forwarded to ``calc_outputs``.

    Returns
    -------
    results
        The algorithm's return value for the configured output.

    Notes
    -----
    Synthetic states use unit wind speed and fixed direction, turbulence
    intensity and air density. They provide deterministic metadata rather
    than replacing supplied scientific results.
    """
    if algo is None:
        if farm is not None:
            farm = deepcopy(farm)
            for turbine in farm.turbines:
                if not turbine.models:
                    turbine.add_model("null_type")
        if farm is None:
            farm = foxes.WindFarm()
            count = 1 if farm_results is None else farm_results.sizes[FC.TURBINE]
            for i in range(count):
                farm.add_turbine(
                    foxes.Turbine(
                        [float(i), 0.0], H=90.0, D=100.0, turbine_models=["null_type"]
                    ),
                    verbosity=0,
                )
        index = (
            [0]
            if farm_results is None or FC.STATE not in farm_results.sizes
            else farm_results[FC.STATE].to_numpy()
        )
        states = foxes.input.states.StatesTable(
            pd.DataFrame({FV.WS: np.ones(len(index))}, index=index),
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            fixed_vars={FV.WD: 270.0, FV.TI: 0.08, FV.RHO: 1.225},
        )
        algo = foxes.algorithms.Downwind(farm, states, wake_models=[], verbosity=0)
    return algo.calc_outputs(
        outputs=output, farm_results=farm_results, point_results=point_results
    )
