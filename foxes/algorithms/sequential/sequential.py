from foxes.algorithms.downwind.downwind import Downwind

import foxes.variables as FV
import foxes.constants as FC
from . import models as mdls

class FarmCalcIter:
    """
    An iterator for farm calculations.
    """
    def __init__(self, algo):
        self.algo = algo

    def __iter__(self):
        """ Initialize the iterator """

        # get models and model data:
        self._mlist, self._calc_pars = self.algo._collect_farm_models(self.algo.calc_pars, self.algo.ambient)
        self._mdata = self.algo.get_models_idata()

        # setup states iterator:
        self._siter = iter(self.algo.states)

        if self.verbosity > 0:
            s = "\n".join([f'  {v}: {d.dtype}, shape {d.shape}' 
                           for v, d in self._mdata['data_vars'].items()])
            self.print("\nInput data:\n\n", s, "\n")
            self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))

    def __next__(self):
        """ Evaluate the next state """
        if self._si < self.states.size():
            si, sind, weight = next(self._siter)
            return si, sind, weight
        else:
            del self._siter, self._mdata, self._mlist, self._calc_pars
            raise StopIteration
        
class Sequential(Downwind):
    """
    A sequential calculation of states without chunking.

    This is of use for the evaluation in simulation
    environments that do not support multi-state computations,
    like FMUs.

    Attributes
    ----------
    ambient: bool
        Flag for ambient calculation
    calc_pars: dict
        Parameters for model calculation.
        Key: model name str, value: parameter dict

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

    def __init__(
            self, 
            mbook, 
            farm, 
            states, 
            *args, 
            ambient=False, 
            calc_pars={},
            **kwargs,
        ):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.ModelBook
            The model book
        farm: foxes.WindFarm
            The wind farm
        states: foxes.core.States
            The ambient states
        args: tuple, optional
            Arguments for Downwind
        ambient: bool
            Flag for ambient calculation
        calc_pars: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        kwargs: dict, optional
            Keyword arguments for Downwind

        """
        super().__init__(
            mbook,
            farm,
            mdls.IterStates(states),
            *args, 
            **kwargs
        )
        self.ambient = ambient
        self.calc_pars = calc_pars

    def farm_calc_iter(self):
        """
        Prepares the iteration.

        Returns
        -------
        iter: FarmCalcIter
            The iterator object

        """

        # get models and model data:
        self._mlist, self._calc_pars = self._collect_farm_models(self.calc_pars, self.ambient)
        self._mdata = self.get_models_idata()

        # setup states iterator:
        self._siter = iter(self.states)

        if self.verbosity > 0:
            s = "\n".join([f'  {v}: {d.dtype}, shape {d.shape}' 
                           for v, d in self._mdata['data_vars'].items()])
            self.print("\nInput data:\n\n", s, "\n")
            self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))

        return self

    def __iter__(self):
        return self.farm_calc_iter()
    


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
