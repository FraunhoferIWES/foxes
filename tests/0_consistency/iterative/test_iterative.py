from pathlib import Path
import inspect

import foxes
import foxes.variables as FV
from foxes.config import config

thisdir = Path(inspect.getfile(inspect.currentframe())).parent


def test():

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
    with foxes.Engine.new("process", chunk_size_states=1000):
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


if __name__ == "__main__":
    test()
