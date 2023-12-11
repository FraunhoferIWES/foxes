from pathlib import Path
import inspect

import foxes
import foxes.variables as FV
import foxes.constants as FC

thisdir = Path(inspect.getfile(inspect.currentframe())).parent


def test():
    c = 1000
    ttype = "DTU10MW"
    sfile = "wind_rose_bremen.csv"
    lfile = thisdir / "test_farm.csv"
    cases = [
        (foxes.algorithms.Downwind, "rotor_wd"),
        (foxes.algorithms.Iterative, "rotor_wd"),
        (foxes.algorithms.Iterative, "rotor_wd_farmo"),
    ]
    lims = {FV.REWS: 5e-7, FV.P: 5e-4}

    ck = {FC.STATE: c}

    base_results = None
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

        algo = Algo(
            mbook,
            farm,
            states=states,
            rotor_model="centre",
            wake_models=["Bastankhah_linear_k002", "IECTI2019_max"],
            wake_frame=frame,
            partial_wakes_model="auto",
            chunks=ck,
            verbosity=1,
        )

        with foxes.utils.runners.DaskRunner() as runner:
            data = runner.run(algo.calc_farm)

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
                chk = delta[v].abs()
                print(f"CASE {(Algo.__name__, frame, v, lim)}:", chk.max())

            assert (chk < lim).all()


if __name__ == "__main__":
    test()
