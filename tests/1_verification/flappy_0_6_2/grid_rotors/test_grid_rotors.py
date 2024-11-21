import pandas as pd
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV
from foxes.config import config

thisdir = Path(inspect.getfile(inspect.currentframe())).parent


def test():
    c = 500
    cpath = thisdir / "flappy"
    tfile = thisdir / "NREL-5MW-D126-H90.csv"
    sfile = thisdir / "states.csv.gz"
    lfile = thisdir / "test_farm.csv"
    cases = [
        (["Basta"], "centre", "rotor_points"),
        (["Basta"], "grid4", "grid4"),
        (["Basta"], "grid16", "grid16"),
        (["Basta"], "grid64", "grid64"),
    ]

    with foxes.Engine.new("threads", chunk_size_states=c):
        for i, (wakes, rotor, pwake) in enumerate(cases):
            print(f"\nENTERING CASE {(wakes, rotor, pwake)}\n")

            mbook = foxes.models.ModelBook()
            ttype = foxes.models.turbine_types.PCtFile(
                data_source=tfile, var_ws_ct=FV.REWS, var_ws_P=FV.REWS
            )
            mbook.turbine_types[ttype.name] = ttype

            mbook.wake_models["Basta"] = foxes.models.wake_models.wind.Bastankhah2014(
                sbeta_factor=0.25, superposition="ws_linear", induction="Betz"
            )

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
                fixed_vars={FV.RHO: 1.225, FV.Z0: 0.1, FV.H: 100.0},
                profiles={FV.WS: "ABLLogNeutralWsProfile"},
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm,
                lfile,
                turbine_models=["kTI_amb_02", ttype.name],
                verbosity=1,
            )

            algo = foxes.algorithms.Downwind(
                farm,
                states,
                mbook=mbook,
                rotor_model=rotor,
                wake_models=wakes,
                wake_frame="rotor_wd",
                partial_wakes=pwake,
                verbosity=1,
            )

            data = algo.calc_farm()

            df = data.to_dataframe()[
                [FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]
            ]

            cfile = cpath / f"results_{i}.csv.gz"
            print("\nReading file", cfile)
            fdata = pd.read_csv(cfile).set_index(["state", "turbine"])

            print()
            print("TRESULTS\n")
            print(df)
            print(fdata)

            print("\nVERIFYING\n")
            df[FV.WS] = df["REWS"]
            df[FV.AMB_WS] = df["AMB_REWS"]

            delta = df - fdata
            print(delta)
            chk = delta[[FV.AMB_WS, FV.AMB_P, FV.WS, FV.P]]
            print(chk)
            chk = chk.abs()
            print(chk.max())

            var = FV.AMB_WS
            sel = chk[var] >= 1e-7
            print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            print(df.loc[sel])
            print(fdata.loc[sel])
            print(delta.loc[sel])
            assert (chk[var] < 1e-7).all()

            var = FV.AMB_P
            sel = chk[var] >= 1e-5
            print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            print(df.loc[sel])
            print(fdata.loc[sel])
            print(delta.loc[sel])
            assert (chk[var] < 1e-5).all()

            var = FV.WS
            sel = chk[var] >= 1.7e-3
            print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            print(df.loc[sel])
            print(fdata.loc[sel])
            print(delta.loc[sel])
            assert (chk[var] < 1.7e-3).all()

            var = FV.P
            sel = chk[var] >= 1.51
            print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            print(df.loc[sel])
            print(fdata.loc[sel])
            print(delta.loc[sel])
            assert (chk[var] < 1.51).all()

            print()


if __name__ == "__main__":
    test()
