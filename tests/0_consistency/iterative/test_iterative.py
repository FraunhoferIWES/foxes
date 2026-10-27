from pathlib import Path
import inspect

import numpy as np
import pandas as pd

import foxes
import foxes.variables as FV


def _run_order_case(algo_type, turbine_types, wake_models, operating=None):
    xy = np.array(
        [(x, y) for y in (0.0, 1100.0, 2200.0) for x in (0.0, 1100.0, 2200.0)]
    )
    wind_directions = np.arange(0.0, 360.0, 10.0)
    mbook = foxes.ModelBook()
    farm_controller = "basic_ctrl"
    if operating is not None:
        mbook.farm_controllers["op"] = foxes.models.farm_controllers.OpFlagController(
            operating
        )
        farm_controller = "op"
    states = foxes.input.states.StatesTable(
        pd.DataFrame(
            {
                "ws": 9.0,
                "wd": wind_directions,
                "ti": 0.06,
            }
        ),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )
    farm = foxes.WindFarm()
    for point, turbine_type in zip(xy, turbine_types):
        farm.add_turbine(
            foxes.Turbine(xy=point, turbine_models=[turbine_type]), verbosity=0
        )
    algo = algo_type(
        farm,
        states,
        wake_models=wake_models,
        farm_controller=farm_controller,
        mbook=mbook,
        verbosity=0,
    )
    with foxes.Engine.new("default", verbosity=0):
        return algo.calc_farm()[FV.P].values


def test_iterative_keeps_turbine_type_order():
    turbine_types = ["DTU10MW", "IEA15MW"] * 4 + ["DTU10MW"]
    wake_models = ["Bastankhah2014_linear_k004"]
    downwind_power = _run_order_case(
        foxes.algorithms.Downwind, turbine_types, wake_models
    )
    iterative_power = _run_order_case(
        foxes.algorithms.Iterative, turbine_types, wake_models
    )

    np.testing.assert_allclose(iterative_power, downwind_power, rtol=1e-6)


def test_iterative_keeps_operating_flag_order():
    operating = np.ones((36, 9), dtype=bool)
    operating[:, 0] = False
    power = _run_order_case(
        foxes.algorithms.Iterative,
        ["DTU10MW"] * 9,
        ["Bastankhah2014_linear_k004", "Rathmann"],
        operating=operating,
    )

    np.testing.assert_array_equal(power == 0.0, ~operating)


def test():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    print("TESTDIR:", thisdir)

    ttype = "DTU10MW"
    sfile = "wind_rose_bremen.csv"
    lfile = thisdir / "test_farm.csv"
    cases = [
        (foxes.algorithms.Downwind, "rotor_wd"),
        (foxes.algorithms.Iterative, "rotor_wd"),
        (foxes.algorithms.Iterative, "rotor_wd_farmo"),
    ]
    lims = {FV.REWS: 5e-7, FV.P: 5e-4}

    base_results = None
    with foxes.Engine.new("threads", chunk_size_states=1000, n_procs=2):
        for Algo, frame in cases:
            print(f"\nENTERING CASE {(Algo.__name__, frame)}\n")

            mbook = foxes.models.ModelBook()

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
                fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm, lfile, turbine_models=[ttype], verbosity=1
            )

            kwargs = {}
            if Algo is foxes.algorithms.Iterative:
                kwargs["mod_cutin"] = {"modify_ct": False, "modify_P": False}

            algo = Algo(
                farm,
                states,
                mbook=mbook,
                rotor_model="grid16",
                wake_models=["Bastankhah2014_linear_k004", "IECTI2019_max"],
                wake_frame=frame,
                partial_wakes="rotor_points",
                verbosity=1,
                **kwargs,
            )

            # f Algo is foxes.algorithms.Iterative:
            #    algo.set_urelax("post_rotor", CT=0.9)

            data = algo.calc_farm()

            df = data.to_dataframe()[
                [FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]
            ]

            print()
            print("TRESULTS\n")
            print(df)

            df = df.reset_index()

            if base_results is None:
                base_results = df

            else:
                print(f"CASE {(Algo.__name__, frame)}")
                delta = df - base_results
                print(delta)
                print(delta.min(), delta.max())

                for v, lim in lims.items():
                    chk = delta[v].abs().loc[df["AMB_REWS"] > 7]
                    print(f"CASE {(Algo.__name__, frame, v, lim)}:", chk.max())

                assert (chk < lim).all()


def test_iterative_max_it_final_run_regression():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent

    ttype = "DTU10MW"
    sfile = "wind_rose_bremen.csv"
    lfile = thisdir / "test_farm.csv"

    mbook = foxes.models.ModelBook()

    states = foxes.input.states.StatesTable(
        data_source=sfile,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.WEIGHT: "weight"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, lfile, turbine_models=[ttype], verbosity=1
    )

    algo = foxes.algorithms.Iterative(
        farm,
        states,
        mbook=mbook,
        rotor_model="grid16",
        wake_models=["Bastankhah2014_linear_k004", "IECTI2019_max"],
        wake_frame="rotor_wd",
        partial_wakes="rotor_points",
        verbosity=0,
        max_it=1,
        mod_cutin={"modify_ct": False, "modify_P": False},
    )

    with foxes.Engine.new("threads", chunk_size_states=1000, n_procs=2):
        results = algo.calc_farm()

    assert results.sizes["state"] == algo.n_states
    assert results.sizes["turbine"] == farm.n_turbines


if __name__ == "__main__":
    test()
