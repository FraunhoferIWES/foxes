import pandas as pd
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV
from foxes.config import config

thisdir = Path(inspect.getfile(inspect.currentframe())).parent


def test():
    print(thisdir)

    cfile = thisdir / "flappy" / "results.csv.gz"
    tPfile = thisdir / "NREL-5MW-D126-H90-P.csv"
    tCtfile = thisdir / "NREL-5MW-D126-H90-Ct.csv"
    sfile = thisdir / "states.csv.gz"
    lfile = thisdir / "test_farm.csv"

    mbook = foxes.models.ModelBook()
    ttype = foxes.models.turbine_types.PCtFromTwo(
        data_source_P=tPfile,
        data_source_ct=tCtfile,
        col_ws_P_file="ws",
        col_ws_ct_file="ws",
        col_P="P",
        col_ct="ct",
        P_nominal=5000,
        H=90.0,
        D=126,
        name="power_and_ct_curve",
    )
    mbook.turbine_types[ttype.name] = ttype

    states = foxes.input.states.StatesTable(
        data_source=sfile,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
        fixed_vars={FV.RHO: 1.225},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, lfile, turbine_models=[ttype.name], verbosity=0
    )

    with foxes.Engine.new("threads", chunk_size_states=1000):

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

        df = data.to_dataframe()[
            [FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]
        ]
        df = df.reset_index()

        print()
        print("TRESULTS\n")
        print(df)

        # print("\Reading file", cfile)
        fdata = pd.read_csv(cfile)
        print(fdata)

        print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        # neglecting ws < 5 and ws > 20
        sel_ws = (
            (fdata[FV.WS] > 5)
            & (fdata[FV.WS] < 20)
            & (df["REWS"] > 5)
            & (df["REWS"] < 20)
        )

        # calculating difference
        delta = df.reset_index() - fdata
        delta = delta[sel_ws]
        print(delta)
        print(delta.max())
        chk = delta[[FV.AMB_WS, FV.AMB_P, FV.WS, FV.P]].abs()
        sel = chk[FV.WS] >= 1e-5
        print(sel)
        print(df[sel & sel_ws])
        print(fdata[sel & sel_ws])
        print(chk.loc[sel & sel_ws])
        print(chk.max())

        assert ((chk[FV.WS] < 1e-5)).all()
        assert (chk[FV.P] < 1e-3).all()


if __name__ == "__main__":
    test()
