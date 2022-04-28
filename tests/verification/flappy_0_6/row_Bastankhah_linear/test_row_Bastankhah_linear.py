
import numpy as np
import pandas as pd
import dask
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
        rotor = "grid9"
        s     = "threads"
        c     = 100
        p0    = np.array([0., 0.])
        stp   = np.array([533., 12.])
        cfile = self.thisdir / "flappy" / "results.csv.gz"
        tfile = self.thisdir / "toyTurbine.csv"

        with dask.config.set(scheduler=s):

            ck = {FV.STATE: c}

            mbook = foxes.models.ModelBook()
            mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                            name="TOYT", filepath=tfile, 
                                            D=120., H=100.)

            states = foxes.input.states.ScanWS(
                ws_list=np.linspace(3., 30., n_s),
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
                turbine_models=["kTI_02", "TOYT"],
                verbosity=self.verbosity
            )
            
            algo = foxes.algorithms.Downwind(
                        mbook,
                        farm,
                        states=states,
                        rotor_model=rotor,
                        turbine_order="order_wd",
                        wake_models=['Bastankhah_linear'],
                        wake_frame="mean_wd",
                        partial_wakes_model="rotor_points",
                        chunks=ck,
                        verbosity=self.verbosity
                    )
            
            data = algo.calc_farm()

        df = data.to_dataframe()[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

        self.print()
        self.print("TRESULTS\n")
        self.print(df)

        self.print("\Reading file", cfile)
        fdata = pd.read_csv(cfile).set_index(["state", "turbine"])
        self.print(fdata)

        self.print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        delta = df - fdata
        delta = delta.loc[delta[FV.AMB_WS] > 3.]
        self.print(delta)
        chk = delta[[FV.AMB_WS, FV.AMB_P, FV.WS, FV.P]]
        self.print(chk)
        chk = chk.abs()
        self.print(chk.max())

        var = FV.WS
        sel = chk[var] >= 1e-5
        self.print(f"\nCHECKING {var}\n", delta.loc[sel])
        assert(chk.loc[sel, var].all())

        var = FV.P
        sel = chk[var] >= 1e-3
        self.print(f"\nCHECKING {var}\n", delta.loc[sel])
        assert(chk.loc[sel, var].all())
        
        

if __name__ == '__main__':
    unittest.main()