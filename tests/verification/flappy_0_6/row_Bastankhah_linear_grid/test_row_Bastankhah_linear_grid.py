
import numpy as np
import pandas as pd
import unittest
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV

class Test(unittest.TestCase):

    def setUp(self):
        self.thisdir   = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 0

    def print(self, *args):
        if self.verbosity:
            print(*args)

    def test(self):
            
        n_s   = 99
        n_t   = 84
        wd    = 88.1
        ti    = 0.04
        rotor = "centre"
        pwake = "grid9"
        c     = 100
        p0    = np.array([0., 0.])
        stp   = np.array([533., 12.])
        cfile = self.thisdir / "flappy" / "results.csv.gz"
        tfile = self.thisdir / "NREL-5MW-D126-H90.csv"

        ck = {FV.STATE: c}

        mbook = foxes.models.ModelBook()
        ttype = foxes.models.turbine_types.PCtFile(data_source=tfile, 
                                        var_ws_ct=FV.REWS, var_ws_P=FV.REWS)
        mbook.turbine_types[ttype.name] = ttype

        states = foxes.input.states.ScanWS(
            ws_list=np.linspace(6., 16., n_s),
            wd=wd,
            ti=ti,
            rho=1.225
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=p0, 
            xy_step=stp, 
            n_turbines=n_t,
            turbine_models=["kTI_amb_02", ttype.name],
            verbosity=self.verbosity
        )
        
        algo = foxes.algorithms.Downwind(
                    mbook,
                    farm,
                    states=states,
                    rotor_model=rotor,
                    turbine_order="order_wd",
                    wake_models=['Bastankhah_linear'],
                    wake_frame="rotor_wd",
                    partial_wakes_model=pwake,
                    chunks=ck,
                    verbosity=self.verbosity
                )
        
        data = algo.calc_farm()

        df = data.to_dataframe()[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

        self.print("\Reading file", cfile)
        fdata = pd.read_csv(cfile).set_index(["state", "turbine"])

        self.print()
        self.print("TRESULTS\n")
        #sel   = (df[FV.P]>0) & (fdata[FV.P]>0)
        #df    = df.loc[sel]
        #fdata = fdata.loc[sel]
        self.print(df)
        self.print(fdata)

        self.print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        delta = df - fdata
        self.print(delta)

        chk = delta.abs()
        self.print(chk.max())

        var = FV.WS
        self.print(f"\nCHECKING {var}")
        sel = chk[var] >= 1e-7
        self.print(df.loc[sel])
        self.print(fdata.loc[sel])
        self.print(chk.loc[sel])
        assert((chk[var] < 1e-7 ).all())

        var = FV.P
        self.print(f"\nCHECKING {var}")
        sel = chk[var] >= 1e-5
        self.print(df.loc[sel])
        self.print(fdata.loc[sel])
        self.print(chk.loc[sel])
        assert((chk[var] < 1e-5).all())
        

if __name__ == '__main__':
    unittest.main()