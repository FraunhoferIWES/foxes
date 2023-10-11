from foxes.algorithms.downwind.downwind import Downwind

import foxes.variables as FV
import foxes.constants as FC
from . import models as mdls
       
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
            mdls.DummyStates(states),
            *args, 
            chunks=None,
            **kwargs
        )
        self.ambient = ambient
        self.calc_pars = calc_pars
        self.keep_models.add(self.states.name)
    
    def iter(self, *args, **kwargs):
        """
        Get a cusomized iterator
        
        Parameters
        ----------
        args: tuple, optional
            Additional arguments for the constructor
        kwargs: dict, optional
            Additional arguments for the constructor
        
        """
        return iter(mdls.SequentialIter(self, *args, **kwargs))

    def __iter__(self):
        """ Get the default iterator """
        return self.iter()
