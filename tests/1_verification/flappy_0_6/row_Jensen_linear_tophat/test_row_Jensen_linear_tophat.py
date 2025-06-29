import numpy as np
import pandas as pd
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV


def test():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    print("TESTDIR:", thisdir)

    n_s = 1000
    n_t = 55
    c = 1000
    p0 = np.array([0.0, 0.0])
    stp = np.array([500.0, 80.0])
    cfile = thisdir / "flappy" / "results.csv.gz"
    tfile = thisdir / "NREL-5MW-D126-H90.csv"

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFile(
        data_source=tfile, var_ws_ct=FV.REWS, var_ws_P=FV.REWS
    )
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.ScanStates(
        {
            FV.WS: np.linspace(3.0, 30.0, n_s),
            FV.WD: [270],
            FV.TI: [0.08],
            FV.RHO: [1.225],
        }
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0,
        xy_step=stp,
        n_turbines=n_t,
        turbine_models=[ttype.name],
        verbosity=0,
    )

    with foxes.Engine.new("threads", chunk_size_states=c):
        algo = foxes.algorithms.Downwind(
            farm,
            states,
            mbook=mbook,
            rotor_model="centre",
            wake_models=["Jensen_linear_k007"],
            wake_frame="rotor_wd",
            partial_wakes={"Jensen_linear_k007": "top_hat"},
            verbosity=0,
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


if __name__ == "__main__":
    test()
