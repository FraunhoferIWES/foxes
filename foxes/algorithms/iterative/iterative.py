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
    kwargs : dict, optional
        Keyword arguments for the Downwind algorithm

    Attributes
    ----------
    conv : foxes.algorithm.iterative.convergence.ConvCrit
        The convergence criteria

    """

    FarmWakesCalculation = FarmWakesCalculation

    def __init__(self, *args, conv=DefaultConv(), **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = conv

    def _collect_farm_models(
            self,
            vars_to_amb,
            init_parameters,
            calc_parameters,
            final_parameters,
            clear_mem_models,
            ambient,
        ):
        """
        Helper function that creates model list
        """
        mlist, init_pars, calc_pars, final_pars = super()._collect_farm_models(
            vars_to_amb, init_parameters, calc_parameters, 
            final_parameters, clear_mem_models, ambient)

        mlist = LoopRunner(self.conv, mlist.models)
        mlist.model_wflag[-1] = True

        return mlist, init_pars, calc_pars, final_pars
