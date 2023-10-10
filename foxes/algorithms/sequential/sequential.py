from foxes.algorithms.downwind.downwind import Downwind

import foxes.variables as FV
from . import models as mdls


class Sequential(Downwind):
    """
    A sequential calculation of states without chunking.

    This is of use for the evaluation in simulation
    environments that do not support multi-state computations,
    like FMUs.

    :group: algorithms.sequential

    """

    @classmethod
    def get_model(cls, name):
        """
        Get the algorithm specific model
        
        Parameters
        ----------
        name: str
            The model name
        
        Returns
        -------
        model: foxes.core.model
            The model
        
        """
        try:
            return getattr(mdls, name)
        except AttributeError:
            return super().get_model(name)

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Arguments for Downwind
        kwargs: dict, optional
            Keyword arguments for Downwind

        """
        super().__init__(*args, **kwargs)
    
    def __iter__(self):
        """ Initialize use as iterator """
        self._iters = mdls.IterStates(self.states)
        self._iters.initialize(self, self.verbosity)
        self._siter = iter(self._iters)
        return self
    
    def __next__(self):
        """ Evaluate the next state """
        if self._si < self.states.size():
            si, sind, weight = next(self._siter)
            return si, sind, weight
        else:
            self._iters.finalize(self, self.verbosity)
            del self._iters, self._siter
            raise StopIteration

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """Helper function for running the main farm calculation"""
        self.print(
            f"\nCalculating {self.n_states} states for {self.n_turbines} turbines"
        )
        farm_results = mlist.run_calculation(
            self, *data, out_vars=self.farm_vars, **kwargs
        )
        farm_results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
        if FV.ORDER in farm_results:
            farm_results[FV.ORDER] = farm_results[FV.ORDER].astype(FC.ITYPE)

        return farm_results
