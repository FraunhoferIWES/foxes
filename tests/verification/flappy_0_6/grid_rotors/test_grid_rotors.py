
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
            
        c     = 500
        cpath = self.thisdir / "flappy"
        tfile = self.thisdir / "NREL-5MW-D126-H90.csv"
        sfile = self.thisdir / "states.csv.gz"
        lfile = self.thisdir / "test_farm.csv"
        cases = [
            (['Bastankhah_linear'], "centre", "rotor_points"), 
            (['Bastankhah_linear'], "grid4", "grid4"),
            (['Bastankhah_linear'], "grid16", "grid16"),
            (['Bastankhah_linear'], "grid64", "grid64")
        ]

        ck = {FV.STATE: c}

        for i, (wakes, rotor, pwake) in enumerate(cases):

            self.print(f"\nENTERING CASE {(wakes, rotor, pwake)}\n")

            mbook = foxes.models.ModelBook()
            ttype = foxes.models.turbine_types.PCtFile(data_source=tfile, 
                                            var_ws_ct=FV.REWS, var_ws_P=FV.REWS)
            mbook.turbine_types[ttype.name] = ttype

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
                fixed_vars={FV.RHO: 1.225, FV.Z0: 0.1, FV.H: 100.},
                profiles={FV.WS: "ABLLogNeutralWsProfile"}
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm,
                lfile,
                turbine_models=["kTI_amb_02", ttype.name],
                verbosity=self.verbosity
            )
            
            algo = foxes.algorithms.Downwind(
                        mbook,
                        farm,
                        states=states,
                        rotor_model=rotor,
                        wake_models=wakes,
                        wake_frame="rotor_wd",
                        partial_wakes_model=pwake,
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
            sel = chk[var] >= 1e-7
            self.print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            self.print(df.loc[sel])
            self.print(fdata.loc[sel])
            self.print(delta.loc[sel])
            assert((chk[var] < 1e-7).all())

            var = FV.AMB_P
            sel = chk[var] >= 1e-5
            self.print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            self.print(df.loc[sel])
            self.print(fdata.loc[sel])
            self.print(delta.loc[sel])
            assert((chk[var] < 1e-5).all())

            var = FV.WS
            sel = chk[var] >= 1.7e-3
            self.print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            self.print(df.loc[sel])
            self.print(fdata.loc[sel])
            self.print(delta.loc[sel])
            assert((chk[var] < 1.7e-3).all())

            var = FV.P
            sel = chk[var] >= 1.51
            self.print(f"\nCHECKING {var}, {(wakes, rotor, pwake)}\n")
            self.print(df.loc[sel])
            self.print(fdata.loc[sel])
            self.print(delta.loc[sel])
            assert((chk[var] < 1.51).all())
        
            self.print()
            
        

if __name__ == '__main__':
    unittest.main()