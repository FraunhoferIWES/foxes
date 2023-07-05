from foxes.algorithms.downwind.downwind import Downwind

from foxes.core import FarmDataModelList
import foxes.variables as FV
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

        verbosity=int(kwargs.pop("verbosity", 1)) #- 1
        super().__init__(*args, verbosity=verbosity, **kwargs)

        self.max_it = 2*self.farm.n_turbines if max_it is None else max_it
        self.conv_crit = DefaultConv() if conv_crit is None else conv_crit
        self._it = None
    
    @property
    def iterations(self):
        """
        The current iteration number
        
        Returns
        -------
        it: int
            The current iteration number
            
        """
        return self._it

    def _collect_farm_models(
        self,
        vars_to_amb,
        calc_parameters,
        ambient,
    ):
        """
        Helper function that creates model list
        """
        
        if self._it == 0:

            out = super()._collect_farm_models(
                vars_to_amb, calc_parameters,ambient)
            
            mdls = [
                self.states,
                self.rotor_model,
                self.farm_controller,
                self.wake_frame,
                self.partial_wakes_model,
            ] + self.wake_models

            self.keep_models.update([self.name, f"{self.name}_calc", "calc_wakes"])
            for m in mdls:
                m.keep(self)

            return out

        # prepare:
        calc_pars = []
        mlist = FarmDataModelList(models=[])
        mlist.name = f"{self.name}_calc"

        # add model that calculates wake effects:
        if not ambient:
            mlist.models.append(self.FarmWakesCalculation())
            mlist.models[-1].name = "calc_wakes"
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # initialize models:
        self.update_idata(mlist)
        
        # update variables:
        self.farm_vars = [FV.WEIGHT, FV.ORDER] + mlist.output_farm_vars(self)

        return mlist, calc_pars

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """ Helper function for running the main farm calculation """
        ir = None if self.prev_farm_results is None \
            else self.chunked(self.prev_farm_results)
        return super()._run_farm_calc(mlist, *data, initial_results=ir, **kwargs)
    
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
        self._it = -1
        while self._it < self.max_it:

            self._it += 1

            self.print("\nALGO: IT =", self._it, vlim=0)

            self.prev_farm_results = fres
            fres = super().calc_farm(finalize=False, **kwargs)

            conv = self.conv_crit.check_converged(self, self.prev_farm_results, fres, 
                                              verbosity=self.verbosity)
            
            if self._it > 0 and self.verbosity >= 0:
                self.print("\nALGO: Convergence results", vlim=0)
                for v, d in self.conv_crit.get_deltas().items():
                    self.print(f"  Delta {v}: {d:.6f}", vlim=0)
            
            if conv:
                self.print("\nALGO: Converged.\n", vlim=0)
                break

        return fres
