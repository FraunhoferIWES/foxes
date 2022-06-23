
import unittest
from pathlib import Path
import inspect
from dask.diagnostics import ProgressBar
#import dask
#from dask.distributed import Client

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
            
        c     = 180
        tfile = self.thisdir / "NREL-5MW-D126-H90.csv"
        sfile = self.thisdir / "states.csv.gz"
        lfile = self.thisdir / "test_farm.csv"
        cases = [
            ("grid100", "rotor_points", None),
            ("grid4", "rotor_points", 0.15),
            ("grid9", "rotor_points", 0.06),
            ("centre", "axiwake5", 0.04),
            ("centre", "axiwake10", 0.038),
            ("grid9", "distsliced", 0.06),
            ("centre", "distsliced9", 0.06),
            ("centre", "distsliced16", 0.04),
            ("centre", "distsliced36", 0.016),
            ("centre", "grid9", 0.06),
            ("centre", "grid16", 0.04),
            ("centre", "grid36", 0.016)
        ]

        ck = {FV.STATE: c}
        #client = Client()
        #self.print(f"\n{client}")
        #self.print(f"Dashboard: {client.dashboard_link}\n")
        #dask.config.set(scheduler="distributed")

        base_results = None
        for rotor, pwake, lim in cases:

            self.print(f"\nENTERING CASE {(rotor, pwake, lim)}\n")

            mbook = foxes.models.ModelBook()
            ttype = foxes.models.turbine_types.PCtFile(data_source=tfile, 
                                            var_ws_ct=FV.REWS, var_ws_P=FV.REWS)
            mbook.turbine_types[ttype.name] = ttype

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
                fixed_vars={FV.RHO: 1.225}
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm,
                lfile,
                turbine_models=[ttype.name],
                verbosity=self.verbosity
            )

            algo = foxes.algorithms.Downwind(
                        mbook,
                        farm,
                        states=states,
                        rotor_model=rotor,
                        wake_models=['Bastankhah_linear_k002'],
                        wake_frame="rotor_wd",
                        partial_wakes_model=pwake,
                        chunks=ck,
                        verbosity=self.verbosity
                    )
            
            if self.verbosity:
                with ProgressBar():
                    data = algo.calc_farm()
            else:
                data = algo.calc_farm()

            df = data.to_dataframe()[[FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]]

            self.print()
            self.print("TRESULTS\n")
            self.print(df)

            df = df.reset_index()

            if base_results is None:
                base_results = df
            
            else:
                self.print(f"CASE {(rotor, pwake, lim)}")
                delta = df - base_results
                self.print(delta)
                self.print(delta.min(), delta.max())
                chk = delta[FV.REWS].abs()
                self.print(f"CASE {(rotor, pwake, lim)}:", chk.max())
            
                assert((chk < lim).all())
        
        

if __name__ == '__main__':
    unittest.main()