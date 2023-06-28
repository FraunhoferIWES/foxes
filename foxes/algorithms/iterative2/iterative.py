from foxes.algorithms.downwind.downwind import Downwind

class Iterative2(Downwind):

    def __init__(self, max_it=100, **kwargs):
        super().__init__(**kwargs)
        self.max_it = max_it

    def calc_farm(
        self,
        **kwargs
    ):
        
        self.prev_farm_results = None
        fres = None
        it = 0
        while it < self.max_it:

            print("\n\nALGO: IT =", it)

            self.prev_farm_results = fres
            fres = super().calc_farm(**kwargs)

            it += 1

            break

        return fres

