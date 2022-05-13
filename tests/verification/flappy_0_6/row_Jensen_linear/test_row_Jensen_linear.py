
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
            
        n_s   = 1000
        n_t   = 55
        c     = 1000
        p0    = np.array([0., 0.])
        stp   = np.array([500., 0.])
        cfile = self.thisdir / "flappy" / "results.csv.gz"
        tfile = self.thisdir / "toyTurbine.csv"

        ck = {FV.STATE: c}

        mbook = foxes.models.ModelBook()
        mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                        name="TOYT", filepath=tfile, 
                                        D=120., H=100.)

        states = foxes.input.states.ScanWS(
            ws_list=np.linspace(3., 30., n_s),
            wd=270.,
            ti=0.08,
            rho=1.225
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=p0, 
            xy_step=stp, 
            n_turbines=n_t,
            turbine_models=["TOYT"],
            verbosity=0
        )
        
        algo = foxes.algorithms.Downwind(
                    mbook,
                    farm,
                    states=states,
                    rotor_model="centre",
                    turbine_order="order_wd",
                    wake_models=['Jensen_linear_k007'],
                    wake_frame="mean_wd",
                    partial_wakes_model="rotor_points",
                    chunks=ck,
                    verbosity=0
                )
        
        data = algo.calc_farm()

        df = data.to_dataframe()[[FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

        self.print()
        self.print("TRESULTS\n")
        self.print(df)

        self.print("\Reading file", cfile)
        fdata = pd.read_csv(cfile)
        self.print(fdata)

        self.print("\nVERIFYING\n")
        df[FV.WS] = df["REWS"]
        df[FV.AMB_WS] = df["AMB_REWS"]

        delta = df.reset_index() - fdata
        self.print(delta.max())
        chk = delta[[FV.WS, FV.P]].max()
        self.print(chk)
        chk = chk.abs()

        assert((chk[FV.WS] < 1e-5).all())
        assert((chk[FV.P] < 1e-3).all())
        
        

if __name__ == '__main__':
    unittest.main()