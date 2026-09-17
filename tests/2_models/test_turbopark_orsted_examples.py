from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.core import TurbineType


def _orsted_turbine_type() -> TurbineType:
    ws = np.arange(0.0, 25.5, 0.5)
    power = [
        0,
        0,
        0,
        0,
        5,
        15,
        37,
        73,
        122,
        183,
        259,
        357,
        477,
        622,
        791,
        988,
        1212,
        1469,
        1755,
        2009,
        2176,
        2298,
        2388,
        2447,
        2485,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        2500,
        0,
    ]
    ct = [
        0,
        0,
        0,
        0,
        0.78,
        0.77,
        0.78,
        0.78,
        0.77,
        0.77,
        0.78,
        0.78,
        0.78,
        0.78,
        0.78,
        0.78,
        0.77,
        0.77,
        0.77,
        0.76,
        0.73,
        0.7,
        0.68,
        0.52,
        0.42,
        0.36,
        0.31,
        0.27,
        0.24,
        0.22,
        0.19,
        0.18,
        0.16,
        0.14,
        0.13,
        0.12,
        0.11,
        0.1,
        0.09,
        0.08,
        0.08,
        0.08,
        0.07,
        0.07,
        0.06,
        0.06,
        0.06,
        0.05,
        0.05,
        0.05,
        0.04,
    ]
    return foxes.models.turbine_types.PCtFile(
        pd.DataFrame({"ws": ws, "P": power, "ct": ct}),
        name="orsted_120m",
        D=120.0,
        H=100.0,
        P_unit="kW",
        rho=None,
    )


def _run_orsted_example_one() -> tuple[np.ndarray, np.ndarray]:
    mbook = foxes.models.ModelBook()
    turbine_type = _orsted_turbine_type()
    mbook.turbine_types[turbine_type.name] = turbine_type
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {
                "ws": np.array([6.0, 10.0, 14.0]) * (100.0 / 90.0) ** 0.1,
                "wd": [270.0, 270.0, 270.0],
                "ti": [0.09, 0.10, 0.11],
            }
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for x in np.arange(4) * 120.0 * 6.0:
        farm.add_turbine(
            foxes.Turbine(xy=[x, 0.0], turbine_models=[turbine_type.name]),
            verbosity=0,
        )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["TurbOPark"],
        rotor_model="centre",
        partial_wakes="gaussian_lookup",
        ground_models={"TurbOPark": "ground_mirror"},
        mbook=mbook,
        verbosity=0,
    )
    with foxes.Engine.new("single", verbosity=0):
        farm_results = algo.calc_farm()
    return farm_results[FV.REWS].to_numpy(), farm_results[FV.P].to_numpy()


def _run_orsted_example_two() -> tuple[np.ndarray, np.ndarray]:
    ws = np.arange(27.0)
    power = [
        0,
        0,
        0,
        0,
        54,
        144,
        289,
        474,
        730,
        1050,
        1417,
        1780,
        2041,
        2199,
        2260,
        2292,
        2299,
        2300,
        2300,
        2300,
        2300,
        2300,
        2300,
        2300,
        2300,
        2300,
        0,
    ]
    ct = [
        0,
        0,
        0,
        0,
        0.94,
        0.82,
        0.76,
        0.68,
        0.86,
        0.83,
        0.77,
        0.68,
        0.66,
        0.52,
        0.47,
        0.41,
        0.38,
        0.34,
        0.27,
        0.26,
        0.23,
        0.22,
        0.22,
        0.2,
        0.16,
        0.17,
        0,
    ]
    mbook = foxes.models.ModelBook()
    turbine_type = foxes.models.turbine_types.PCtFile(
        pd.DataFrame({"ws": ws, "P": power, "ct": ct}),
        name="orsted_80m",
        D=80.0,
        H=70.0,
        P_unit="kW",
        rho=None,
    )
    mbook.turbine_types[turbine_type.name] = turbine_type
    turbine_x = np.arange(4) * 120.0 * 6.0
    speedup = (1.0 + (turbine_x - 5.0) ** 2 * 1.0e-8) / (1.0 + 25.0e-8)
    ambient_ws = (
        np.array([6.0, 10.0, 14.0])[:, None] * (70.0 / 90.0) ** 0.1 * speedup[None, :]
    )
    states = foxes.input.states.TurbinePointCloud(
        xr.Dataset(
            coords={FC.STATE: np.arange(3), FC.TURBINE: np.arange(4)},
            data_vars={
                "ws": ((FC.STATE, FC.TURBINE), ambient_ws),
                "wd": ((FC.STATE, FC.TURBINE), np.full((3, 4), 270.0)),
                "ti": (
                    (FC.STATE, FC.TURBINE),
                    np.broadcast_to([0.09, 0.10, 0.11], (4, 3)).T,
                ),
            },
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2ncvar={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for x in turbine_x:
        farm.add_turbine(
            foxes.Turbine(xy=[x, 0.0], turbine_models=[turbine_type.name]),
            verbosity=0,
        )

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["TurbOPark"],
        rotor_model="centre",
        partial_wakes="gaussian_lookup",
        ground_models={"TurbOPark": "ground_mirror"},
        mbook=mbook,
        verbosity=0,
    )
    with foxes.Engine.new("single", verbosity=0):
        farm_results = algo.calc_farm()
    return farm_results[FV.REWS].to_numpy(), farm_results[FV.P].to_numpy()


def test_turbopark_matches_orsted_example_one_single_row():
    rews, power = _run_orsted_example_one()
    expected_rews = np.array(
        [
            [6.06355050721975, 10.1059175120329, 14.1482845168461],
            [4.35804371995363, 7.37377085289107, 12.8635577397301],
            [3.82923165036237, 6.45010832482725, 12.0276667404546],
            [3.48366810406203, 5.87189558198255, 11.1721433436644],
        ]
    ).T
    expected_power = np.array(
        [
            [495.429647093727, 2201.84387293603, 2500.0],
            [165.681333834342, 938.265716039082, 2500.0],
            [105.264701735512, 607.531414199904, 2485.83000221364],
            [71.8241034924664, 446.254939675813, 2408.31291455241],
        ]
    ).T

    np.testing.assert_allclose(rews, expected_rews, rtol=3.0e-3)
    np.testing.assert_allclose(power, expected_power, rtol=3.0e-3)


def test_turbopark_matches_orsted_example_two_single_row():
    rews, power = _run_orsted_example_two()
    expected_rews = np.array(
        [
            [5.85109033776109, 9.75181722960182, 13.6525441214426],
            [4.37635141095725, 7.29654364961552, 11.3076935877073],
            [4.02595954168878, 6.71637563917382, 10.0998726719234],
            [3.94419598700442, 6.48890311114917, 9.45281987051811],
        ]
    ).T
    expected_power = np.array(
        [
            [267.408098975358, 1325.91692326387, 2238.805191408],
            [87.8716269861529, 549.915174301573, 1860.30802639162],
            [56.3363587519906, 421.529493247156, 1453.25377990821],
            [50.9865832982385, 379.447075562596, 1216.18489248015],
        ]
    ).T

    np.testing.assert_allclose(rews, expected_rews, rtol=3.0e-3)
    np.testing.assert_allclose(power, expected_power, rtol=3.0e-3)
