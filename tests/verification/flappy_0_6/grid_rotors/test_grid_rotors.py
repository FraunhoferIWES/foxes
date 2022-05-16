
import pandas as pd
import unittest
from pathlib import Path
import inspect

import foxes
import foxes.variables as FV

class Test(unittest.TestCase):

    def setUp(self):
        self.thisdir   = Path(inspect.getfile(inspect.currentframe())).parent
        self.verbosity = 1

    def print(self, *args):
        if self.verbosity:
            print(*args)

    def test(self):
            
        c     = 500
        cpath = self.thisdir / "flappy"
        tfile = self.thisdir / "toyTurbine.csv"
        sfile = self.thisdir / "states.csv.gz"
        lfile = self.thisdir / "test_farm.csv"
        cases = [
            (['Bastankhah_linear'], "centre"), 
            (['Bastankhah_linear'], "grid4"),
            (['Bastankhah_linear'], "grid16"),
            (['Bastankhah_linear'], "grid64")
        ]

        ck = {FV.STATE: c}

        for i, (wakes, rotor) in enumerate(cases):

            self.print(f"\nENTERING CASE {(wakes, rotor)}\n")

            mbook = foxes.models.ModelBook()
            mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                            name="TOYT", filepath=tfile, D=120., H=100.,
                                            var_ws_ct=FV.REWS, var_ws_P=FV.REWS)

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
                fixed_vars={FV.RHO: 1.225, FV.Z0: 0.1, FV.H: 100.},
                profiles={FV.WS: "ABLLogNeutralWsProfile"},
                verbosity=self.verbosity
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm,
                lfile,
                turbine_models=["kTI_02", "TOYT"],
                verbosity=self.verbosity
            )
            
            algo = foxes.algorithms.Downwind(
                        mbook,
                        farm,
                        states=states,
                        rotor_model=rotor,
                        turbine_order="order_wd",
                        wake_models=wakes,
                        wake_frame="mean_wd",
                        partial_wakes_model="rotor_points",
                        chunks=ck,
                        verbosity=self.verbosity
                    )
            
            data = algo.calc_farm()

            df = data.to_dataframe()[[FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

            cfile = cpath / f"results_{i}.csv.gz"
            self.print("\nReading file", cfile)
            fdata = pd.read_csv(cfile).set_index(["state", "turbine"])

            self.print()
            self.print("TRESULTS\n")
            sel = (df[FV.P] > 0) & (fdata[FV.P]>0)
            df = df.loc[sel]
            fdata = fdata.loc[sel]
            self.print(df)
            self.print(fdata)

            self.print("\nVERIFYING\n")
            df[FV.WS] = df["REWS"]
            df[FV.AMB_WS] = df["AMB_REWS"]

            delta = df - fdata
            self.print(delta)
            chk = delta[[FV.AMB_WS, FV.AMB_P, FV.WS, FV.P]]
            self.print(chk)
            chk = chk.abs()
            self.print(chk.max())

            var = FV.AMB_WS
            sel = chk[var] >= 1e-5
            self.print(f"\nCHECKING {var}\n", delta.loc[sel])
            assert(chk.loc[~sel, var].all())

            var = FV.AMB_P
            sel = chk[var] >= 1e-3
            self.print(f"\nCHECKING {var}\n", delta.loc[sel])
            assert(chk.loc[~sel, var].all())

            var = FV.WS
            sel = chk[var] >= 1e-5
            self.print(f"\nCHECKING {var}\n", delta.loc[sel])
            assert(chk.loc[~sel, var].all())

            var = FV.P
            sel = chk[var] >= 1e-3
            self.print(f"\nCHECKING {var}\n", delta.loc[sel])
            assert(chk.loc[~sel, var].all())
        
            self.print()
            
        

if __name__ == '__main__':
    unittest.main()