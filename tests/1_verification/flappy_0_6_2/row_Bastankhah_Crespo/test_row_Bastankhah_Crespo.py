import numpy as np
import pandas as pd
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV


def test():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    print("TESTDIR:", thisdir)

    n_s = 30
    n_t = 52
    wd = 270.0
    ti = 0.08
    rotor = "centre"
    c = 100
    p0 = np.array([0.0, 0.0])
    stp = np.array([601.0, 15.0])
    cfile = thisdir / "flappy" / "results.csv.gz"
    tfile = thisdir / "NREL-5MW-D126-H90.csv"

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(
        data_source=tfile, var_ws_ct=FV.REWS, var_ws_P=FV.REWS
    )
    mbook.turbine_types[ttype.name] = ttype

    mbook.wake_models["Basta"] = foxes.models.wake_models.wind.Bastankhah2014(
        sbeta_factor=0.25, superposition="ws_quadratic", induction="Betz"
    )
    mbook.wake_models["Crespo"] = foxes.models.wake_models.ti.CrespoHernandezTIWake(
        superposition="ti_max", induction="Betz"
    )

    states = foxes.input.states.ScanStates(
        {
            FV.WS: np.linspace(6.0, 16.0, n_s),
            FV.WD: [wd],
            FV.TI: [ti],
            FV.RHO: [1.225],
        }
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0,
        xy_step=stp,
        n_turbines=n_t,
        turbine_models=["kTI_amb_02", ttype.name],
        verbosity=1,
    )

    with foxes.Engine.new("threads", chunk_size_states=c):
        algo = foxes.algorithms.Downwind(
            farm,
            states,
            mbook=mbook,
            rotor_model=rotor,
            wake_models=["Basta", "Crespo"],
            wake_frame="rotor_wd",
            partial_wakes=["axiwake6", "top_hat"],
            verbosity=1,
        )

        data = algo.calc_farm()

        df = data.to_dataframe()[
            [FV.X, FV.Y, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_TI, FV.TI]
        ]

        print()
        print("TRESULTS\n")
        print(df)

        print("\nReading file", cfile)
        fdata = pd.read_csv(cfile).set_index(["state", "turbine"])

        print()
        print("TRESULTS\n")
        # sel   = (df[FV.P]>0) & (fdata[FV.P]>0)
        # df    = df.loc[sel]
        # fdata = fdata.loc[sel]
        print(df)
        print(fdata)

        print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        delta = df - fdata
        print(df)
        print(delta[[FV.WS, FV.TI]])

        chk = delta.abs()
        print(chk.max())

        var = FV.WS
        print(f"\nCHECKING {var}")
        sel = chk[var] >= 3e-3
        print(df.loc[sel])
        print(fdata.loc[sel])
        print(chk.loc[sel])
        assert (chk[var] < 3e-3).all()

        var = FV.TI
        print(f"\nCHECKING {var}")
        sel = chk[var] >= 3e-4
        print(df.loc[sel])
        print(fdata.loc[sel])
        print(chk.loc[sel])
        assert (chk[var] < 3e-4).all()


if __name__ == "__main__":
    test()
