from foxes.algorithms.downwind.downwind import Downwind

from foxes.core import FarmDataModelList
from foxes.utils import Dict
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
    prev_farm_results: xarray.Dataset
        Results from the previous iteration

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
        super().__init__(*args, **kwargs)

        self.max_it = 2 * self.farm.n_turbines if max_it is None else max_it
        self.conv_crit = (
            self.get_model("DefaultConv")() if conv_crit is None else conv_crit
        )
        self.prev_farm_results = None
        self._it = None
        self._mlist = None
        self._reamb = False
        self._urelax = Dict(
            first={}, pre_rotor={}, post_rotor={}, pre_wake={FV.CT: 0.5}, last={}
        )

    def set_urelax(self, entry_point, **urel):
        """
        Sets under-relaxation parameters.

        Parameters
        ----------
        entry_point: str
            The entry point: first, pre_rotor, post_rotor,
            pre_wake, last
        urel: dict
            The variables and their under-relaxation values

        """
        if self.initialized:
            raise ValueError(f"Attempt to set_urelax after initialization")
        self._urelax[entry_point].update(urel)

    @property
    def urelax(self):
        """
        Returns the under-relaxation parameters

        Returns
        -------
        urlx: foxes.utils.Dict
            The under-relaxation parameters

        """
        return self._urelax

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
            self._mlist0, self._calc_pars0 = super()._collect_farm_models(
                calc_parameters,
                ambient=ambient,
            )

            n = 0
            if len(self._urelax["first"]):
                self._mlist0.insert(0, mdls.URelax(**self._urelax["first"]))
                self._calc_pars0.insert(0, {})
                self._reamb = True
                n += 1

            if len(self._urelax["pre_rotor"]):
                self._mlist0.insert(2 + n, mdls.URelax(**self._urelax["pre_rotor"]))
                self._calc_pars0.insert(2 + n, {})
                self._reamb = True
                n += 1

            if len(self._urelax["post_rotor"]):
                self._mlist0.insert(4 + n, mdls.URelax(**self._urelax["post_rotor"]))
                self._calc_pars0.insert(4 + n, {})
                self._reamb = True
                n += 1

            if len(self._urelax["pre_wake"]):
                self._mlist0.models[7 + n].urelax = mdls.URelax(
                    **self._urelax["pre_wake"]
                )

            if len(self._urelax["last"]):
                self._mlist0.append(mdls.URelax(**self._urelax["last"]))
                self._calc_pars0.append({})

            return self._mlist0, self._calc_pars0

        elif ambient or self._reamb:
            return self._mlist0, self._calc_pars0

        else:
            # prepare:
            calc_pars = []
            mlist = FarmDataModelList(models=[])

            # add under-relaxation during wake calculation:
            urelax = None
            if len(self._urelax["pre_wake"]):
                urelax = mdls.URelax(**self._urelax["pre_wake"])

            # add model that calculates wake effects:
            mlist.models.append(self.get_model("FarmWakesCalculation")(urelax=urelax))
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

            # add under-relaxation:
            if len(self._urelax["last"]):
                mlist.append(mdls.URelax(**self._urelax["last"]))
                calc_pars.append({})

            return mlist, calc_pars

    def _calc_farm_vars(self, mlist):
        """Helper function that gathers the farm variables"""
        if self._it == 0:
            super()._calc_farm_vars(mlist)

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

            if self._it == 0:
                self.verbosity -= 1

        # finalize models:
        if finalize:
            self.print("\n")
            self.finalize()
            for m in self._mlist0.models:
                if m not in self.all_models():
                    m.finalize(self, self.verbosity)
            del self._mlist0, self._calc_pars0

        return fres
