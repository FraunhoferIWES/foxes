from foxes.algorithms.downwind.downwind import Downwind

from foxes.core import FarmDataModelList
import foxes.variables as FV
from . import models as mdls


class Iterative(Downwind):
    """
    Iterative calculation of farm data.

    Attributes
    ----------
    max_it: int
        The maximal number of iterations
    conv_crit: foxes.algorithms.iterative.ConvCrit
        The convergence criteria

    :group: algorithms.iterative

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

        verbosity = int(kwargs.pop("verbosity", 1)) - 1
        super().__init__(*args, verbosity=verbosity, **kwargs)

        self.max_it = 2 * self.farm.n_turbines if max_it is None else max_it
        self.conv_crit = self.get_model("DefaultConv")() if conv_crit is None else conv_crit
        self._it = None
        self._mlist = None

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
        calc_parameters,
        ambient,
    ):
        """
        Helper function that creates model list
        """

        if self._it == 0:
            mlist, calc_pars = super()._collect_farm_models(
                calc_parameters, ambient=False
            )

            # calc_pars[mlist.models.index(self.rotor_model)].update(
            #    {"store_rpoints": True, "store_rweights": True, "store_amb_res": True}
            # )

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

            return mlist, calc_pars

        # prepare:
        calc_pars = []
        mlist = FarmDataModelList(models=[])
        mlist.name = f"{self.name}_calc"

        # add model that calculates wake effects:
        mlist.models.append(self.get_model("FarmWakesCalculation")())
        mlist.models[-1].name = "calc_wakes"
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # initialize models:
        self.update_idata(mlist)

        # update variables:
        self.farm_vars = [
            FV.X,
            FV.Y,
            FV.H,
            FV.D,
            FV.WEIGHT,
            FV.ORDER,
            FV.WD,
            FV.YAW,
        ] + mlist.output_farm_vars(self)
        self.farm_vars += [FV.var2amb[v] for v in self.farm_vars if v in FV.var2amb]
        self.farm_vars = sorted(list(set(self.farm_vars)))

        self._mlist = mlist

        return mlist, calc_pars

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """Helper function for running the main farm calculation"""
        ir = (
            None
            if self.prev_farm_results is None
            else self.chunked(self.prev_farm_results)
        )
        return super()._run_farm_calc(mlist, *data, initial_results=ir, **kwargs)

    def calc_farm(self, finalize=True, **kwargs):
        """
        Calculate farm data.

        Parameters
        ----------
        finalize: bool
            Flag for finalization after calculation
        kwargs: dict, optional
            Arguments for calc_farm in the base class.

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        fres = None
        self._it = -1
        while self._it < self.max_it:
            self._it += 1

            self.print(f"\nAlgorithm {self.name}: Iteration {self._it}\n", vlim=0)

            self.prev_farm_results = fres
            fres = super().calc_farm(finalize=False, **kwargs)

            conv = self.conv_crit.check_converged(
                self, self.prev_farm_results, fres, verbosity=self.verbosity + 1
            )

            if conv:
                self.print(f"\nAlgorithm {self.name}: Convergence reached.\n", vlim=0)
                break

        # finalize models:
        if finalize:
            self.print("\n")
            self.finalize_model(self._mlist)
            self.finalize()
        self.cleanup()
        self._mlist = None

        return fres
