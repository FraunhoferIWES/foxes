from foxes.algorithms.downwind.downwind import Downwind

from .convergence import DefaultConv

class Iterative(Downwind):
    """
    Iterative calculation of farm data.
    
    Attributes
    ----------
    max_it: int
        The maximal number of iterations
    conv_crit: foxes.algorithms.iterative.ConvCrit
        The convergence criteria

    """

    def __init__(self, *args, max_it=None, conv_crit=None, **kwargs):
        """
        Constructor.
        
        Parameters
        ----------
        args: tuple, optional
            Arguments for Downwind
        max_it: int, optional
            The maximal number of iterations
        conv_crit: foxes.algorithms.iterative.ConvCrit, optional
            The convergence criteria
        kwargs: dict, optional
            Keyword arguments for Downwind
        
        """

        verbosity=int(kwargs.pop("verbosity", 1)) - 1
        super().__init__(*args, verbosity=verbosity, **kwargs)

        self.max_it = 2*self.farm.n_turbines if max_it is None else max_it
        self.conv_crit = DefaultConv() if conv_crit is None else conv_crit

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """ Helper function for running the main farm calculation """
        return super()._run_farm_calc(mlist, *data, **kwargs,
                                      initial_results=self.prev_farm_results)
    
    def calc_farm(
        self,
        **kwargs
    ):
        """
        Calculate farm data.

        Parameters
        ----------
        kwargs: dict, optional
            Arguments for calc_farm in the base class.

        Returns
        -------
        farm_results : xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        fres = None
        it = 0
        while it < self.max_it:

            self.print("\nALGO: IT =", it, vlim=-1)

            self.prev_farm_results = fres
            fres = super().calc_farm(**kwargs)

            it += 1

            if self.conv_crit.check_converged(self, self.prev_farm_results, fres, 
                                              verbosity=self.verbosity):
                self.print("\nALGO: Converged.\n", vlim=-1)
                break

        return fres
