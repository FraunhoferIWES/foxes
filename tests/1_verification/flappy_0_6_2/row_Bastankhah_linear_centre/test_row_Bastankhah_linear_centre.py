import numpy as np
import pandas as pd
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV


def test():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    print("TESTDIR:", thisdir)

    n_s = 99
    n_t = 84
    wd = 88.1
    ti = 0.04
    rotor = "centre"
    c = 100
    p0 = np.array([0.0, 0.0])
    stp = np.array([533.0, 12.0])
    cfile = thisdir / "flappy" / "results.csv.gz"
    tfile = thisdir / "NREL-5MW-D126-H90.csv"

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(
        data_source=tfile, var_ws_ct=FV.REWS, var_ws_P=FV.REWS
    )
    mbook.turbine_types[ttype.name] = ttype

    mbook.wake_models["Basta"] = foxes.models.wake_models.wind.Bastankhah2014(
        sbeta_factor=0.25, superposition="ws_linear", induction="Betz"
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
            wake_models=["Basta"],
            wake_frame="rotor_wd",
            partial_wakes="rotor_points",
            verbosity=1,
        )

        data = algo.calc_farm()

        df = data.to_dataframe()[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

        print("\nReading file", cfile)
        fdata = pd.read_csv(cfile).set_index(["state", "turbine"])

        print()
        print("TRESULTS\n")
        sel = (df[FV.P] > 0) & (fdata[FV.P] > 0)
        df = df.loc[sel]
        fdata = fdata.loc[sel]
        print(df.loc[sel])
        print(fdata.loc[sel])

        print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        delta = df - fdata
        print(delta)

        chk = delta.abs()
        print(chk.max())

        var = FV.WS
        print(f"\nCHECKING {var}")
        sel = chk[var] >= 1e-7
        print(df.loc[sel])
        print(fdata.loc[sel])
        print(chk.loc[sel])
        assert (chk[var] < 1e-7).all()

        var = FV.P
        sel = chk[var] >= 1e-5
        print(f"\nCHECKING {var}\n", delta.loc[sel])
        assert (chk[var] < 1e-5).all()
