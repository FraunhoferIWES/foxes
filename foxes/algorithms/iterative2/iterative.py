from foxes.algorithms.downwind.downwind import Downwind
from foxes.core import Data
import foxes.constants as FC

class Iterative2(Downwind):

    def __init__(self, *args, max_it=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_it = max_it

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """ Helper function for running the main farm calculation """
        return super()._run_farm_calc(mlist, *data, **kwargs,
                                      initial_results=self.prev_farm_results)
    
    def calc_farm(
        self,
        **kwargs
    ):
        fres = None
        it = 0
        while it < self.max_it:

            print("\n\nALGO: IT =", it)

            """
            if fres is not None:
                self.prev_farm_results = Data(
                    data={v: d.to_numpy() for v, d in fres.data_vars.items()},
                    dims={v: d.to_numpy() for v, d in fres.dims.items()},
                    loop_dims=[FC.STATE],
                    name="prev_farm_results"
                )
            """
            self.prev_farm_results = fres
            fres = super().calc_farm(**kwargs)

            it += 1

            if it > self.n_turbines:
                break

        return fres

