from foxes.algorithms.downwind.downwind import Downwind
from .models import DefaultConv, LoopRunner, FarmWakesCalculation


class Iterative(Downwind):
    """
    Iterative algorithm that repeats the
    downwind calculation until convergence
    has been achieved

    Parameters
    ----------
    args : tuple, optional
        Arguments for the Downwind algorithm
    conv : foxes.algorithm.iterative.models.ConvCrit
        The convergence criteria
    max_its : int, optional
        Set the maximal number of iterations, None means
        number of turbines + 1
    conv_error : bool
        Throw error if not converging
    kwargs : dict, optional
        Keyword arguments for the Downwind algorithm

    Attributes
    ----------
    conv : foxes.algorithm.iterative.convergence.ConvCrit
        The convergence criteria
    max_its : int
        Set the maximal number of iterations, None means
        number of turbines + 1
    conv_error : bool
        Throw error if not converging

    """

    FarmWakesCalculation = FarmWakesCalculation

    def __init__(
            self, 
            *args, 
            conv=DefaultConv(), 
            max_its=None, 
            conv_error=True,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.conv = conv
        self.max_its = max_its
        self.conv_error = conv_error

    def _collect_farm_models(
            self,
            vars_to_amb,
            calc_parameters,
            ambient,
        ):
        """
        Helper function that creates model list
        """
        # get models from Downwind algorithm:
        mlist, calc_pars = super()._collect_farm_models(
            vars_to_amb, calc_parameters, ambient)

        # wrap the models into a loop:
        mlist = LoopRunner(self.conv, mlist.models, max_its=self.max_its, 
                            conv_error=self.conv_error, verbosity=self.verbosity-1)

        # flag only the last model as wake relevant:
        mlist.model_wflag[-1] = True 

        return mlist, calc_pars
