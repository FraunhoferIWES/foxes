
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
            
        c     = 2000
        cfile = self.thisdir / "flappy" / "results.csv.gz"
        tfile = self.thisdir / "toyTurbine.csv"
        sfile = self.thisdir / "states.csv.gz"
        lfile = self.thisdir / "test_farm.csv"

        ck = {FV.STATE: c}

        mbook = foxes.models.ModelBook()
        mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                        name="TOYT", filepath=tfile, D=120., H=100.)

        states = foxes.input.states.StatesTable(
            data_source=sfile,
            output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
            var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
            fixed_vars={FV.RHO: 1.225},
            verbosity=0
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_from_file(
            farm,
            lfile,
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
                    partial_wakes_model="top_hat",
                    chunks=ck,
                    verbosity=0
                )
        
        data = algo.calc_farm()

        df = data.to_dataframe()[[FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

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
        self.print(delta)
        self.print(delta.max())
        chk = delta[[FV.AMB_WS, FV.AMB_P, FV.WS, FV.P]].abs().max()
        self.print(chk)
        
        assert((chk[FV.WS] < 1e-5).all())
        assert((chk[FV.P] < 1e-3).all())
        
        

if __name__ == '__main__':
    unittest.main()